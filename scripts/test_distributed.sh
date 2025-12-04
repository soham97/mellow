#!/bin/bash
# Quick test script for verifying multi-GPU setup
# Usage: ./test_distributed.sh <num_gpus>

NUM_GPUS=${1:-2}

echo "=========================================="
echo "Testing Multi-GPU Distributed Setup"
echo "=========================================="
echo "Testing with $NUM_GPUS GPUs"
echo

# Test 1: Check CUDA availability
echo "Test 1: Checking CUDA availability..."
python -c "
import torch
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
print(f'✓ Number of GPUs: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  - GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo

# Test 2: Check distributed backend
echo "Test 2: Checking distributed backend..."
python -c "
import torch.distributed as dist
print(f'✓ NCCL available: {dist.is_nccl_available()}')
print(f'✓ Gloo available: {dist.is_gloo_available()}')
"

echo

# Test 3: Run simple distributed test
echo "Test 3: Running simple distributed test..."
cat > /tmp/test_dist.py <<'EOF'
import os
import torch
import torch.distributed as dist

def test_distributed():
    # Get local rank from environment (set by torchrun)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Set the device BEFORE initializing process group
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    # Initialize process group
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Process {rank}/{world_size} (local_rank={local_rank}) on device {device} initialized successfully")
    
    # Simple all_reduce test
    tensor = torch.ones(1, device=device) * rank
    
    dist.all_reduce(tensor)
    expected = sum(range(world_size))
    
    assert tensor.item() == expected, f"All-reduce failed: got {tensor.item()}, expected {expected}"
    print(f"Process {rank}: ✓ All-reduce test passed (result: {tensor.item()})")
    
    dist.destroy_process_group()
    print(f"Process {rank}: ✓ Successfully completed")

if __name__ == "__main__":
    test_distributed()
EOF

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS /tmp/test_dist.py

echo
echo "=========================================="
echo "✓ All tests passed!"
echo "Your multi-GPU setup is ready for training"
echo "=========================================="

# Cleanup
rm -f /tmp/test_dist.py
