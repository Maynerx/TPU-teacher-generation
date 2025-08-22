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


def tensor_to_fixed_array(t: torch.Tensor, dtype=None):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().contiguous().numpy()

    if t.ndim == 3:
        B, L, K = t.shape
        inner = pa.FixedSizeListArray.from_arrays(pa.array(t.reshape(-1), type=dtype), K)
        return pa.FixedSizeListArray.from_arrays(inner, L)
    elif t.ndim == 2:
        B, L = t.shape
        return pa.FixedSizeListArray.from_arrays(pa.array(t.ravel(), type=dtype), L)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {t.shape}")

def _buffer_to_arrow(batch_buffer):
    """
    Convert list of (enc, dec, lab, idx, prob) to a single Arrow table using FixedSizeListArray.
    """

    # Concatenate along batch dimension
    enc = torch.cat([b[0] for b in batch_buffer], dim=0)
    dec = torch.cat([b[1] for b in batch_buffer], dim=0)
    lab = torch.cat([b[2] for b in batch_buffer], dim=0)
    idx = torch.cat([b[3] for b in batch_buffer], dim=0)
    prob = torch.cat([b[4] for b in batch_buffer], dim=0)

    # Create FixedSizeListArrays
    arrow_batch = pa.table({
        "encoder_input": tensor_to_fixed_array(enc),
        "decoder_input": tensor_to_fixed_array(dec),
        "labels":        tensor_to_fixed_array(lab),
        "top_k_indices": tensor_to_fixed_array(idx),
        "top_k_probs":   tensor_to_fixed_array(prob, dtype=pa.float16()),  # cast dtype
    })

    return pa.table(arrow_batch)


def run_process(rank, config):
    def inference_loop(model, data_loader, device, rank, verbose=False, top_k=50, temp=2.0, train=True, write_every_n_batches=256):
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
            enc  = batch["student_encoder_input_ids"].cpu()
            dec  = batch["student_decoder_input_ids"].cpu()
            lab  = batch["student_labels"].cpu()
            idx  = topk_idx.cpu()
            prob = topk_probs.cpu().to(torch.float16)
    
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
