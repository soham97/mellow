#!/bin/bash
#SBATCH -vv
#SBATCH --job-name=mellow_train
#SBATCH -N 2                         # Number of nodes
#SBATCH --gpus=v100-32:16             # Request 16 V100-32GB GPUs total (4 per node)
#SBATCH -t 48:00:00                  # Time limit hrs:min:sec
#SBATCH --output=logs/slurm-%j.out   # Standard output log
#SBATCH --error=logs/slurm-%j.err    # Standard error log
#SBATCH -p GPU                # GPU partition
#SBATCH --export=ALL
#SBATCH --signal INT@60

# Load necessary modules (adjust based on your cluster)
# module load cuda/11.8
# module load python/3.9
# module load pytorch/2.0

# Create logs directory if it doesn't exist
mkdir -p logs

# Get the master node address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Print some debug info
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"

# Activate conda environment
# Update 'qa_gen_3.1' to your conda environment name
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qa_gen_3.1

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Run the training script
srun python train.py \
    --config config/train_4gpu.yaml \
    --distributed-backend nccl \
    --save-dir outputs

# Alternative: Use torchrun (recommended for PyTorch 1.10+)
# This automatically handles SLURM environment variables
# srun python -m torch.distributed.run \
#     --nnodes=$SLURM_NNODES \
#     --nproc_per_node=$SLURM_GPUS_PER_NODE \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     train.py \
#     --config config/train_4gpu.yaml \
#     --distributed-backend nccl \
#     --save-dir outputs
