#!/bin/bash
#SBATCH -vv
#SBATCH --job-name=mellow_debug
#SBATCH -N 1                         # Single node
#SBATCH --gpus=v100-32:2             # Request 2 V100-32GB GPUs for quick testing
#SBATCH -t 1:00:00
#SBATCH --output=logs/debug-%j.out
#SBATCH --error=logs/debug-%j.err
#SBATCH -p GPU-shared                # GPU-shared partition
#SBATCH --export=ALL
#SBATCH --signal INT@60

# Create logs directory
mkdir -p logs

# Set master address and port
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Enable NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Print all SLURM environment variables
echo "=== SLURM Environment Variables ==="
env | grep SLURM | sort
echo "===================================="

echo ""
echo "=== Master Info ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "==================="

echo ""
echo "=== GPU Info ==="
srun --ntasks=1 nvidia-smi
echo "================"

# Activate conda environment
# Update 'qa_gen_3.1' to your conda environment name
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qa_gen_3.1

# Verify environment
echo ""
echo "=== Environment Info ==="
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "========================"

# Run training with debug output
srun python train.py \
    --config config/local3.yaml \
    --distributed-backend nccl \
    --save-dir outputs/debug \
    --functiontest
