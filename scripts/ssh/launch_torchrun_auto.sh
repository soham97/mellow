#!/bin/bash
# Automated multi-node launcher using torchrun via SSH
# Usage: ./launch_torchrun_auto.sh <config_file> <num_gpus_per_node> <node1> <node2> ...
# Example: ./launch_torchrun_auto.sh config/local3.yaml 4 node1 node2 node3

CONFIG_FILE=${1:-"config/local3.yaml"}
GPUS_PER_NODE=${2:-4}
shift 2
NODES=("$@")

if [ ${#NODES[@]} -eq 0 ]; then
    echo "Error: No nodes specified!"
    echo "Usage: $0 <config_file> <num_gpus_per_node> <node1> <node2> ..."
    exit 1
fi

NUM_NODES=${#NODES[@]}
MASTER_ADDR=${NODES[0]}
MASTER_PORT=${MASTER_PORT:-29500}

echo "=========================================="
echo "Automated Multi-Node Training Launcher"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Number of nodes: $NUM_NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Nodes: ${NODES[@]}"
echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "=========================================="

# Conda environment setup commands
CONDA_SETUP="source \$(conda info --base)/etc/profile.d/conda.sh && conda activate qa_gen_3.1"

# Get current directory
WORK_DIR=$(pwd)

# Launch on each node
for NODE_RANK in "${!NODES[@]}"; do
    NODE=${NODES[$NODE_RANK]}
    echo "Launching on node: $NODE (rank: $NODE_RANK)"
    
    if [ "$NODE" == "localhost" ] || [ "$NODE" == "$(hostname)" ]; then
        # Local node
        eval "$CONDA_SETUP"
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
            --save-dir outputs 2>&1 | tee logs/node_${NODE_RANK}.log &
    else
        # Remote node via SSH
        ssh $NODE "cd $WORK_DIR && \
            $CONDA_SETUP && \
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
                --save-dir outputs" 2>&1 | tee logs/node_${NODE_RANK}.log &
    fi
    
    sleep 2  # Small delay between launches
done

echo "=========================================="
echo "All nodes launched. Check logs/node_*.log for output"
echo "Press Ctrl+C to stop all processes"
echo "=========================================="

# Wait for all background processes
wait
