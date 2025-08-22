import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla
from torch_xla.distributed.parallel_loader import ParallelLoader
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
import tqdm.auto as tqdm
import pyarrow as pa
import pyarrow.ipc as ipc
    
import pyarrow.parquet as pq
from time import sleep
import pyarrow.feather as feather
import threading
from queue import Queue

def _buffer_to_arrow(batch_buffer):
    """
    Convert list of (enc, dec, lab, idx, prob) to a single Arrow table using FixedSizeListArray.
    """
    import pyarrow as pa

    # Concatenate along batch dimension
    enc = torch.cat([b[0] for b in batch_buffer], dim=0)
    dec = torch.cat([b[1] for b in batch_buffer], dim=0)
    lab = torch.cat([b[2] for b in batch_buffer], dim=0)
    idx = torch.cat([b[3] for b in batch_buffer], dim=0)
    prob = torch.cat([b[4] for b in batch_buffer], dim=0)

    # Create FixedSizeListArrays
    batch_cpu = {
        "encoder_input": pa.FixedSizeListArray.from_arrays(pa.array(enc.view(-1)), enc.shape[1]),
        "decoder_input": pa.FixedSizeListArray.from_arrays(pa.array(dec.view(-1)), dec.shape[1]),
        "labels":        pa.FixedSizeListArray.from_arrays(pa.array(lab.view(-1)), lab.shape[1]),
        "top_k_indices": pa.FixedSizeListArray.from_arrays(pa.array(idx.view(-1)), idx.shape[1]),
        "top_k_probs":   pa.FixedSizeListArray.from_arrays(pa.array(prob.view(-1)), prob.shape[1]),
    }

    return pa.table(batch_cpu)


def run_process(rank, config):
    def inference_loop(model, data_loader, device, rank, verbose=False, top_k=50, temp=2.0, train=True, write_every_n_batches=16):
        """
        TPU-friendly inference loop writing to Arrow IPC files (faster than Parquet).
        """
        
        model.eval()
        data_loader = tqdm.tqdm(data_loader) if rank == 0 else data_loader
        batch_buffer = []
    
        for batch_idx, batch in enumerate(data_loader):
            input_ids_t = batch["teacher_input_ids"].to(device).long()
            attn_mask_t = batch["teacher_attention_mask"].to(device)
    
            with torch.no_grad():
                logits = model(input_ids=input_ids_t, attention_mask=attn_mask_t).logits
                topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
                topk_probs = F.softmax(topk_vals / temp, dim=-1)
            xm.mark_step()  # trigger XLA execution
    
            # Move tensors to CPU once
            enc  = batch["student_encoder_input_ids"].cpu().numpy()
            dec  = batch["student_decoder_input_ids"].cpu().numpy()
            lab  = batch["student_labels"].cpu().numpy()
            idx  = topk_idx.cpu().numpy()
            prob = topk_probs.cpu().to(torch.float16).numpy()
    
            batch_buffer.append((enc, dec, lab, idx, prob))
    
            # Write every N batches
            if len(batch_buffer) >= write_every_n_batches:
                arrow_table = _buffer_to_arrow(batch_buffer)
                filename = f"{'train' if train else 'val'}/shard_{rank}_{batch_idx}.arrow"
                with ipc.new_file(filename, arrow_table.schema) as writer:
                    writer.write_table(arrow_table)
                batch_buffer.clear()
    
        # Write remaining batches
        if batch_buffer:
            arrow_table = _buffer_to_arrow(batch_buffer)
            filename = f"{'train' if train else 'val'}/shard_{rank}_final.arrow"
            with ipc.new_file(filename, arrow_table.schema) as writer:
                writer.write_table(arrow_table)
                    
    
    sleep(rank * 1.5)
    if rank == 0:
        print("Generation starting...")
    device = xm.xla_device()

    trainiing_dataset = config.training_dataset
    validation_dataset = config.validation_dataset

    trainiing_dataset = trainiing_dataset.shard(index=rank, num_shards=torch_xla.runtime.world_size())
    validation_dataset = validation_dataset.shard(index=rank, num_shards=torch_xla.runtime.world_size())

    training_loader = DataLoader(
        trainiing_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=True,
    )

    xla_loader = ParallelLoader(training_loader, [device]).per_device_loader(device)
    xla_val_loader = ParallelLoader(validation_loader, [device]).per_device_loader(device)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True
    ).cpu()

    model.resize_token_embeddings(len(config.tokenizer), mean_resizing=False)

    model.to(device).eval()
    xm.rendezvous("model_loaded")
    sleep(rank * 1.5)
    if rank == 0:
        print("Loop starting...")
    # Training loop
    inference_loop(
        model,
        xla_loader,
        device,
        rank,
        verbose=True if rank == 0 else False,
        top_k=config.top_k,
        temp=config.temperature,
        train=True
    )

    # Validation loop
    inference_loop(
        model,
        xla_val_loader,
        device,
        rank,
        verbose=True if rank == 0 else False,
        top_k=config.top_k,
        temp=config.temperature,
        train=False
    )
    if rank == 0:
        print("Generation completed.")
    xm.rendezvous("close")
