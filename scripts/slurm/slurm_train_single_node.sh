#!/bin/bash
# Single-node training on SLURM cluster
# Update CONDA_ENV below to match your conda environment name
#SBATCH -vv
#SBATCH --job-name=mellow_train_single
#SBATCH -N 1                         # Single node
#SBATCH --gpus=v100-32:4             # Request 4 V100-32GB GPUs
#SBATCH -t 24:00:00                  # Time limit
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH -p GPU-shared                # GPU partition
#SBATCH --export=ALL
#SBATCH --signal INT@60

# Create logs directory
mkdir -p logs

# Set master address and port
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Print debug info
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"

# Activate conda environment
# Update 'qa_gen_3.1' to your conda environment name
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qa_gen_3.1

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Run training
srun python train.py \
    --config config/train_4gpu.yaml \
    --distributed-backend nccl \
    --save-dir outputs
