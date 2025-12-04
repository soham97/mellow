#!/bin/bash
# Multi-node multi-GPU training launcher for direct SSH access
# Usage: ./launch_multinode.sh <config_file> <num_nodes> <num_gpus_per_node> <node_list>
# Example: ./launch_multinode.sh config/local3.yaml 2 4 "node1,node2"

CONFIG_FILE=${1:-"config/local3.yaml"}
NUM_NODES=${2:-2}
GPUS_PER_NODE=${3:-4}
NODE_LIST=${4:-"localhost"}

# Set master node as the first node in the list
MASTER_NODE=$(echo $NODE_LIST | cut -d',' -f1)
MASTER_PORT=${MASTER_PORT:-29500}

echo "=========================================="
echo "Multi-Node Multi-GPU Training Launcher"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Number of nodes: $NUM_NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Node list: $NODE_LIST"
echo "Master node: $MASTER_NODE"
echo "Master port: $MASTER_PORT"
echo "=========================================="

# Conda environment setup
CONDA_SETUP="source \$(conda info --base)/etc/profile.d/conda.sh && conda activate qa_gen_3.1"

# Export environment variables
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

# Create logs directory
mkdir -p logs

# Convert node list to array
IFS=',' read -ra NODES <<< "$NODE_LIST"

# Launch training on each node
NODE_RANK=0
for NODE in "${NODES[@]}"; do
    echo "Launching on node: $NODE (rank: $NODE_RANK)"
    
    if [ "$NODE" == "localhost" ] || [ "$NODE" == "$(hostname)" ]; then
        # Local node - run in background
        eval "$CONDA_SETUP"
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        MASTER_ADDR=$MASTER_ADDR \
        MASTER_PORT=$MASTER_PORT \
        WORLD_SIZE=$WORLD_SIZE \
        NODE_RANK=$NODE_RANK \
        python -m torch.distributed.launch \
            --nproc_per_node=$GPUS_PER_NODE \
            --nnodes=$NUM_NODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            train.py \
            --config $CONFIG_FILE \
            --distributed-backend nccl \
            --save-dir outputs &
    else
        # Remote node - launch via SSH
        ssh $NODE "cd $(pwd) && \
            $CONDA_SETUP && \
            CUDA_VISIBLE_DEVICES=0,1,2,3 \
            MASTER_ADDR=$MASTER_ADDR \
            MASTER_PORT=$MASTER_PORT \
            WORLD_SIZE=$WORLD_SIZE \
            NODE_RANK=$NODE_RANK \
            python -m torch.distributed.launch \
                --nproc_per_node=$GPUS_PER_NODE \
                --nnodes=$NUM_NODES \
                --node_rank=$NODE_RANK \
                --master_addr=$MASTER_ADDR \
                --master_port=$MASTER_PORT \
                train.py \
                --config $CONFIG_FILE \
                --distributed-backend nccl \
                --save-dir outputs" &
    fi
    
    NODE_RANK=$((NODE_RANK + 1))
    sleep 2  # Small delay between node launches
done

echo "=========================================="
echo "All nodes launched. Training in progress..."
echo "Press Ctrl+C to stop all processes"
echo "=========================================="

# Wait for all background processes
wait
