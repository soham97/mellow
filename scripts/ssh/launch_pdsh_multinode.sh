#!/bin/bash
# Multi-node launcher using pdsh (Parallel Distributed Shell)
# Requires: pdsh to be installed on the system
# Usage: ./launch_pdsh_multinode.sh <config_file> <num_gpus_per_node> <node_list>
# Example: ./launch_pdsh_multinode.sh config/local3.yaml 4 "node[1-4]"

CONFIG_FILE=${1:-"config/local3.yaml"}
GPUS_PER_NODE=${2:-4}
NODE_LIST=${3:-"localhost"}

# Check if pdsh is installed
if ! command -v pdsh &> /dev/null; then
    echo "Error: pdsh is not installed!"
    echo "Install with: sudo apt-get install pdsh  (Ubuntu/Debian)"
    echo "          or: sudo yum install pdsh      (CentOS/RHEL)"
    exit 1
fi

# Extract master node
MASTER_NODE=$(echo $NODE_LIST | sed 's/\[.*\]//' | sed 's/,.*//')
MASTER_PORT=${MASTER_PORT:-29500}

# Count number of nodes
NUM_NODES=$(pdsh -w $NODE_LIST hostname | wc -l)

echo "=========================================="
echo "Multi-Node Training with pdsh"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Number of nodes: $NUM_NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Node pattern: $NODE_LIST"
echo "Master node: $MASTER_NODE"
echo "Master port: $MASTER_PORT"
echo "=========================================="

# Get current directory
WORK_DIR=$(pwd)

# Create logs directory
mkdir -p logs

# Export script to run on each node
cat > /tmp/ddp_launch.sh <<EOF
#!/bin/bash
cd $WORK_DIR

# Activate conda environment
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate qa_gen_3.1

# Get node rank based on hostname
NODE_HOSTNAME=\$(hostname)
NODE_RANK=\$(pdsh -w $NODE_LIST hostname | grep -n \$NODE_HOSTNAME | cut -d: -f1)
NODE_RANK=\$((NODE_RANK - 1))

echo "Node \$NODE_HOSTNAME starting with rank \$NODE_RANK"

torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=\$NODE_RANK \
    --master_addr=$MASTER_NODE \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_NODE:$MASTER_PORT \
    train.py \
    --config $CONFIG_FILE \
    --distributed-backend nccl \
    --save-dir outputs
EOF

chmod +x /tmp/ddp_launch.sh

# Copy launch script to all nodes
pdsh -w $NODE_LIST "mkdir -p /tmp"
pdcp -w $NODE_LIST /tmp/ddp_launch.sh /tmp/ddp_launch.sh

# Launch on all nodes simultaneously
echo "Launching training on all nodes..."
pdsh -w $NODE_LIST "/tmp/ddp_launch.sh" 2>&1 | tee logs/pdsh_training.log

echo "=========================================="
echo "Training complete. Check logs/pdsh_training.log"
echo "=========================================="

# Cleanup
rm -f /tmp/ddp_launch.sh
