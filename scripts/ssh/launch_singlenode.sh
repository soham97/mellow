#!/bin/bash
# Single-node multi-GPU training launcher for direct terminal access
# Update CONDA_ENV in the script below to match your conda environment name
# Usage: ./launch_singlenode.sh <config_file> <num_gpus>
# Example: ./launch_singlenode.sh config/train_4gpu.yaml 4

CONFIG_FILE=${1:-"config/train_4gpu.yaml"}
NUM_GPUS=${2:-4}
MASTER_PORT=${MASTER_PORT:-29500}

echo "=========================================="
echo "Single-Node Multi-GPU Training Launcher"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "Master port: $MASTER_PORT"
echo "=========================================="

# Activate conda environment
# Update 'qa_gen_3.1' to your conda environment name
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qa_gen_3.1

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Launch with torchrun (recommended for PyTorch >= 1.10)
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train.py \
    --config $CONFIG_FILE \
    --distributed-backend nccl \
    --save-dir outputs

# Alternative: Using torch.distributed.launch (older method)
# python -m torch.distributed.launch \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=$MASTER_PORT \
#     train.py \
#     --config $CONFIG_FILE \
#     --distributed-backend nccl \
#     --save-dir outputs
