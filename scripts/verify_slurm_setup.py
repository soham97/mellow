#!/usr/bin/env python3
"""
SLURM DDP Setup Verification Script

Run this script to verify your environment is correctly set up for SLURM-based DDP training.
Can be run interactively or as a SLURM job.

Usage:
    python verify_slurm_setup.py
    
Or in SLURM:
    srun --nodes=1 --ntasks-per-node=2 --gres=gpu:2 python verify_slurm_setup.py
"""

import os
import sys
import socket

def print_section(title):
    """Print a section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_slurm_env():
    """Check SLURM environment variables"""
    print_section("SLURM Environment Variables")
    
    slurm_vars = [
        'SLURM_JOB_ID', 'SLURM_PROCID', 'SLURM_LOCALID', 
        'SLURM_NTASKS', 'SLURM_NTASKS_PER_NODE', 'SLURM_NODELIST',
        'SLURM_GPUS_PER_NODE', 'SLURM_NNODES'
    ]
    
    found_slurm = False
    for var in slurm_vars:
        value = os.environ.get(var, "NOT SET")
        print(f"  {var:25s} = {value}")
        if value != "NOT SET":
            found_slurm = True
    
    if not found_slurm:
        print("\n  ⚠️  WARNING: No SLURM variables detected!")
        print("  Running outside SLURM or SLURM not configured.")
    else:
        print("\n  ✓ SLURM environment detected")
    
    return found_slurm

def check_pytorch():
    """Check PyTorch installation and CUDA availability"""
    print_section("PyTorch Configuration")
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            print("\n  ✓ PyTorch with CUDA is ready")
            return True
        else:
            print("\n  ⚠️  WARNING: CUDA not available!")
            print("  Check your PyTorch installation and CUDA drivers.")
            return False
    except ImportError:
        print("  ❌ ERROR: PyTorch not installed!")
        print("  Install with: pip install torch")
        return False

def check_distributed_env():
    """Check distributed training environment variables"""
    print_section("Distributed Training Environment")
    
    dist_vars = ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK']
    
    for var in dist_vars:
        value = os.environ.get(var, "NOT SET")
        print(f"  {var:15s} = {value}")

def check_nccl():
    """Check NCCL availability"""
    print_section("NCCL Backend")
    
    try:
        import torch
        import torch.distributed as dist
        
        if torch.cuda.is_available():
            has_nccl = dist.is_nccl_available()
            print(f"  NCCL available: {has_nccl}")
            
            if has_nccl:
                print("  ✓ NCCL backend is ready for multi-GPU training")
            else:
                print("  ⚠️  WARNING: NCCL not available!")
                print("  Multi-GPU training may not work optimally.")
        else:
            print("  Skipped (CUDA not available)")
    except Exception as e:
        print(f"  ❌ ERROR checking NCCL: {e}")

def check_network():
    """Check network configuration"""
    print_section("Network Configuration")
    
    hostname = socket.gethostname()
    try:
        ip = socket.gethostbyname(hostname)
        print(f"  Hostname: {hostname}")
        print(f"  IP Address: {ip}")
    except:
        print(f"  Hostname: {hostname}")
        print("  IP Address: Could not resolve")
    
    master_addr = os.environ.get('MASTER_ADDR', 'NOT SET')
    master_port = os.environ.get('MASTER_PORT', 'NOT SET')
    print(f"  MASTER_ADDR: {master_addr}")
    print(f"  MASTER_PORT: {master_port}")

def check_dependencies():
    """Check required Python packages"""
    print_section("Python Dependencies")
    
    required_packages = [
        'torch', 'numpy', 'yaml', 'tqdm', 'pandas', 
        'scipy', 'transformers'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            version = __import__(package).__version__ if hasattr(__import__(package), '__version__') else "unknown"
            print(f"  ✓ {package:20s} (version: {version})")
        except ImportError:
            print(f"  ❌ {package:20s} NOT INSTALLED")
            all_installed = False
    
    if all_installed:
        print("\n  ✓ All required packages are installed")
    else:
        print("\n  ⚠️  Some packages are missing!")
        print("  Install with: pip install -r requirements.txt")

def test_distributed_init():
    """Test distributed initialization"""
    print_section("Testing Distributed Initialization")
    
    # Check if we're in a SLURM environment with distributed setup
    if 'SLURM_PROCID' not in os.environ:
        print("  Skipped (not in SLURM distributed environment)")
        print("  To test distributed initialization, run with srun:")
        print("    srun --nodes=1 --ntasks-per-node=2 --gres=gpu:2 python verify_slurm_setup.py")
        return
    
    try:
        import torch
        import torch.distributed as dist
        
        # Set up environment variables for distributed training
        if 'SLURM_PROCID' in os.environ:
            if 'RANK' not in os.environ:
                os.environ['RANK'] = os.environ['SLURM_PROCID']
            if 'WORLD_SIZE' not in os.environ:
                os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
            if 'LOCAL_RANK' not in os.environ:
                os.environ['LOCAL_RANK'] = os.environ.get('SLURM_LOCALID', '0')
        
        # Set MASTER_ADDR if not set
        if 'MASTER_ADDR' not in os.environ:
            import subprocess
            try:
                nodelist = os.environ.get('SLURM_NODELIST', '')
                if nodelist:
                    result = subprocess.run(f"scontrol show hostnames {nodelist}".split(), 
                                          capture_output=True, text=True)
                    nodes = result.stdout.strip().split('\n')
                    if nodes:
                        os.environ['MASTER_ADDR'] = nodes[0]
            except:
                os.environ['MASTER_ADDR'] = socket.gethostname()
        
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
        
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        print(f"  Rank: {rank}")
        print(f"  World Size: {world_size}")
        print(f"  Backend: nccl")
        
        if torch.cuda.is_available():
            # Initialize process group
            dist.init_process_group(backend='nccl')
            
            print(f"  ✓ Successfully initialized process group")
            print(f"    - Initialized rank: {dist.get_rank()}")
            print(f"    - World size: {dist.get_world_size()}")
            
            # Cleanup
            dist.destroy_process_group()
        else:
            print("  Skipped (CUDA not available)")
            
    except Exception as e:
        print(f"  ❌ ERROR: Failed to initialize distributed training")
        print(f"  Error: {e}")

def main():
    """Main verification function"""
    print("\n" + "🔍 " * 30)
    print("   SLURM DDP Setup Verification")
    print("🔍 " * 30)
    
    has_slurm = check_slurm_env()
    has_cuda = check_pytorch()
    check_distributed_env()
    check_nccl()
    check_network()
    check_dependencies()
    test_distributed_init()
    
    # Summary
    print_section("Summary")
    
    if has_slurm and has_cuda:
        print("  ✓ Environment looks good for SLURM DDP training!")
        print("\n  Next steps:")
        print("    1. Test with: sbatch slurm_debug.sh")
        print("    2. Run training: sbatch slurm_train_single_node.sh")
    elif has_cuda:
        print("  ✓ PyTorch with CUDA is ready")
        print("  ℹ️  Not running in SLURM environment")
        print("\n  To test in SLURM:")
        print("    srun --nodes=1 --ntasks-per-node=2 --gres=gpu:2 python verify_slurm_setup.py")
    else:
        print("  ⚠️  Some issues detected. Please review the output above.")
    
    print("\n")

if __name__ == "__main__":
    main()
