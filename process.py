import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla
from torch_xla.distributed.parallel_loader import ParallelLoader
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
import tqdm.auto as tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from time import sleep



def run_process(rank, config):
    def inference_loop(model, data_loader, device, rank, verbose=False, top_k=50, temp=2.0, train=True):
        model.eval()
        data_loader = tqdm.tqdm(data_loader) if rank == 0 else data_loader
        writer = None
    
        for batch in data_loader:
            input_ids_t = batch["teacher_input_ids"].to(device)           # (B, 2L)
            input_ids_t = input_ids_t.long() 
            attn_mask_t = batch["teacher_attention_mask"].to(device)
            with torch.no_grad():
                logits = model(input_ids=input_ids_t, attention_mask=attn_mask_t).logits        # (B, L, V)
                topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
                topk_probs = F.softmax(topk_vals / temp, dim=-1)
            xm.mark_step()  # compile & run this sub-graph
    
            topk_vals = topk_vals.cpu()
            topk_idx  = topk_idx.cpu()
            topk_probs = topk_probs.cpu()
    
            batch_cpu = {
                "encoder_input": batch["student_encoder_input_ids"].cpu().tolist(),
                "decoder_input": batch["student_decoder_input_ids"].cpu().tolist(),
                "labels":        batch["student_labels"].cpu().tolist(),
                "top_k_indices": topk_idx.cpu().tolist(),
                "top_k_probs":   topk_probs.cpu().tolist(),
            }
    
            arrow_batch = pa.table(batch_cpu)
    
            if writer is None:
                writer = pq.ParquetWriter(
                    f"{'train' if train else 'val'}/shard_{rank}.parquet",
                    schema=arrow_batch.schema,
                    compression="SNAPPY",
                    use_dictionary=True
                )
            writer.write_table(arrow_batch)
    
        if writer is not None:
            writer.close()
            
    
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
