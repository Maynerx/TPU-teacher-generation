import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from data import load_data
from process import run_process
import torch_xla.distributed.xla_multiprocessing as xmp


class Config:
    def __init__(self,
                model_name: str,
                seq_len: int,
                batch_size: int,
                top_k: int,
                temperature: float,
                num_workers: int
                ):
        self.model_name = model_name
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.top_k = top_k
        self.temperature = temperature
        self.training_dataset = None
        self.validation_dataset = None
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def empty_load(model_name):
    _ = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False
    ).to('cpu')
    del _

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return Config(
        model_name=config_data['model_name'],
        seq_len=config_data['seq_len'],
        batch_size=config_data['batch_size'],
        top_k=config_data['top_k'],
        temperature=config_data['temperature'],
        training_dataset=config_data['training_dataset'],
        validation_dataset=config_data['validation_dataset'],
        num_workers=config_data['num_workers']
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config file")
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--use-fork", type=bool, default=False, help="Use fork start method for multiprocessing")
    args = parser.parse_args()

    empty_load(args.model_name)

    config = load_config(args.config)

    tokenizer = config.tokenizer

    training_dataset, validation_dataset, tokenizer = load_data(
        args.data_path,
        tokenizer,
        config.seq_len, 
        ratio=0.1,
    )
    print(f"Training dataset size: {len(training_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    config.training_dataset = training_dataset
    config.validation_dataset = validation_dataset

    os.makedirs('train', exist_ok=True)
    os.makedirs('val', exist_ok=True)

    xmp.spawn(run_process,
              args=(config,),
              start_method='fork' if args.use_fork else 'spawn')


