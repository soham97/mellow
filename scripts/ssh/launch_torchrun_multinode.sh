#!/bin/bash
# Multi-node multi-GPU training with torchrun (PyTorch >= 1.10)
# This script should be run on EACH node separately with appropriate NODE_RANK
# 
# Usage on each node:
#   NODE_RANK=0 ./launch_torchrun_multinode.sh config/local3.yaml 2 4 master_node_ip
#   NODE_RANK=1 ./launch_torchrun_multinode.sh config/local3.yaml 2 4 master_node_ip
#
# Or use the automated version: launch_torchrun_auto.sh

CONFIG_FILE=${1:-"config/local3.yaml"}
NUM_NODES=${2:-2}
GPUS_PER_NODE=${3:-4}
MASTER_ADDR=${4:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}

echo "=========================================="
echo "Multi-Node Training with torchrun"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Number of nodes: $NUM_NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Node rank: $NODE_RANK"
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

# Launch with torchrun
torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --config $CONFIG_FILE \
    --distributed-backend nccl \
    --save-dir outputs
