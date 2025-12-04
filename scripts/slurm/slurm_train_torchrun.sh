#!/bin/bash
#SBATCH -vv
#SBATCH --job-name=mellow_torchrun
#SBATCH -N 2                         # Number of nodes
#SBATCH --gpus=v100-32:16             # Request 16 V100-32GB GPUs total (4 per node)
#SBATCH -t 48:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH -p GPU                # GPU-shared partition
#SBATCH --export=ALL
#SBATCH --signal INT@60

# Create logs directory
mkdir -p logs

# Get node information
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NNODES: $SLURM_NNODES"

# Get number of GPUs per node
GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | awk -F':' '{print $NF}')
if [ -z "$GPUS_PER_NODE" ]; then
    GPUS_PER_NODE=4  # default
fi

echo "GPUS_PER_NODE: $GPUS_PER_NODE"

# Activate conda environment
# Update 'qa_gen_3.1' to your conda environment name
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qa_gen_3.1

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Using torchrun (recommended for PyTorch >= 1.10)
# This automatically sets up all distributed environment variables
srun bash -c "
    python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --config config/train_4gpu.yaml \
    --distributed-backend nccl \
    --save-dir outputs
"
