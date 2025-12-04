#!/usr/bin/env python3
"""
Debug script to check what environment variables are set by torchrun/SLURM
Usage: torchrun --standalone --nnodes=1 --nproc_per_node=4 check_env.py
"""

import os
import sys

def main():
    print(f"\n{'='*60}")
    print(f"Process PID: {os.getpid()}")
    print(f"{'='*60}")
    
    # Check for all distributed-related environment variables
    env_vars = [
        'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE',
        'MASTER_ADDR', 'MASTER_PORT',
        'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NTASKS', 'SLURM_NTASKS_PER_NODE',
        'SLURM_JOB_ID', 'SLURM_NODELIST', 'SLURM_GPUS_PER_NODE',
        'CUDA_VISIBLE_DEVICES', 'TORCHELASTIC_RUN_ID'
    ]
    
    print("\nEnvironment Variables:")
    print("-" * 60)
    for var in env_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"  {var:25s} = {value}")
    
    print("\nAll RANK/SLURM variables:")
    print("-" * 60)
    for key, value in sorted(os.environ.items()):
        if 'RANK' in key or 'SLURM' in key or 'TORCH' in key or 'CUDA' in key:
            print(f"  {key:30s} = {value}")
    
    # Check if PyTorch can see GPUs
    try:
        import torch
        print(f"\n{'='*60}")
        print("PyTorch GPU Info:")
        print("-" * 60)
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device count: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("\nPyTorch not available")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
