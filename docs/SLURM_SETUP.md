# SLURM Multi-GPU DDP Training Setup Guide

This guide explains how to run the Mellow training code on SLURM-based HPC clusters with multi-GPU distributed data parallel (DDP) training.

## Prerequisites

1. **SLURM cluster** with GPU nodes
2. **PyTorch** with CUDA support
3. **NCCL backend** for efficient GPU communication
4. All dependencies from `requirements.txt` installed

## Environment Setup

### 1. Install Dependencies

```bash
# Create conda environment (recommended)
conda create -n mellow python=3.9
conda activate mellow

# Install PyTorch with CUDA support (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### 2. Configure Your Training

Edit your config file (e.g., `config/local3.yaml`) to set:
- `train.num_nodes`: Number of SLURM nodes
- `train.batch_size`: Per-GPU batch size
- Other training hyperparameters

## Running Training on SLURM

### Single Node Training (4 GPUs)

1. **Edit the SLURM script** `slurm_train_single_node.sh`:
   - Adjust `#SBATCH` parameters for your cluster
   - Set correct partition name
   - Adjust memory, CPU cores, and time limits
   - Update config file path

2. **Submit the job**:
   ```bash
   sbatch scripts/slurm/slurm_train_single_node.sh
   ```

3. **Monitor the job**:
   ```bash
   # Check job status
   squeue -u $USER
   
   # View output logs
   tail -f logs/slurm-<job_id>.out
   tail -f logs/slurm-<job_id>.err
   ```

### Multi-Node Training (e.g., 2 nodes x 4 GPUs = 8 GPUs)

1. **Edit the SLURM script** `slurm_train.sh`:
   - Set `#SBATCH --nodes=2` (or your desired number)
   - Set `#SBATCH --ntasks-per-node=4` (GPUs per node)
   - Set `#SBATCH --gres=gpu:4`
   - Adjust other parameters as needed

2. **Submit the job**:
   ```bash
   sbatch scripts/slurm/slurm_train.sh
   ```

## SLURM Environment Variables

The code automatically detects and uses these SLURM environment variables:

- `SLURM_JOB_ID`: Job identifier
- `SLURM_PROCID`: Global rank (0 to world_size-1)
- `SLURM_LOCALID`: Local rank on the node (0 to GPUs_per_node-1)
- `SLURM_NTASKS`: Total number of processes (world size)
- `SLURM_NTASKS_PER_NODE`: Number of tasks per node
- `SLURM_NODELIST`: List of allocated nodes
- `SLURM_GPUS_PER_NODE`: GPUs per node

## Key SLURM Parameters

### Resource Allocation

```bash
#SBATCH --nodes=2                    # Number of nodes
#SBATCH --ntasks-per-node=4          # Tasks per node (usually = GPUs per node)
#SBATCH --gres=gpu:4                 # GPUs per node
#SBATCH --cpus-per-task=8            # CPU cores per GPU
#SBATCH --mem=200GB                  # Total memory per node
```

### Time and Output

```bash
#SBATCH --time=48:00:00              # Max runtime (HH:MM:SS)
#SBATCH --output=logs/slurm-%j.out   # stdout (%j = job ID)
#SBATCH --error=logs/slurm-%j.err    # stderr
```

### Partition Selection

```bash
#SBATCH --partition=gpu              # Queue/partition name
```

## Communication Backend

The code uses **NCCL** backend for GPU communication, which is optimal for:
- Multi-GPU training on single node
- Multi-node training with GPUs
- High bandwidth GPU interconnects (NVLink, InfiniBand)

Specify with: `--distributed-backend nccl`

## Troubleshooting

### 1. Connection Timeout

**Error**: `Connection timeout` or `Unable to initialize distributed`

**Solution**:
- Ensure `MASTER_ADDR` and `MASTER_PORT` are set correctly
- Check firewall rules between nodes
- Verify nodes can communicate: `ping <node_name>`

### 2. NCCL Errors

**Error**: `NCCL error` or `NCCL initialization failed`

**Solution**:
- Load correct NCCL module: `module load nccl/2.x`
- Check GPU visibility: `echo $CUDA_VISIBLE_DEVICES`
- Try setting: `export NCCL_DEBUG=INFO` for detailed logs

### 3. Out of Memory

**Error**: `CUDA out of memory`

**Solution**:
- Reduce batch size in config
- Reduce `--cpus-per-task` if CPU memory is the issue
- Request more memory: `#SBATCH --mem=400GB`

### 4. Module Not Found

**Error**: `ModuleNotFoundError`

**Solution**:
- Activate conda environment in SLURM script
- Install missing packages: `pip install <package>`
- Check Python path: `which python`

### 5. Wrong LOCAL_RANK

**Error**: Rank mismatch or incorrect GPU assignment

**Solution**:
- Ensure `SLURM_LOCALID` is available
- Check `srun` vs `mpirun` usage
- Verify SLURM configuration on your cluster

## Performance Tips

1. **Batch Size**: Scale with number of GPUs
   - Example: 32 per GPU × 8 GPUs = 256 effective batch size

2. **Data Loading**: Use sufficient workers
   - Set `num_workers` in config to `--cpus-per-task - 1`

3. **Mixed Precision**: Enable for faster training
   - Already configured in your code with AMP

4. **Gradient Accumulation**: If batch size is limited
   - Accumulate gradients over multiple steps

5. **Network**: Use high-speed interconnects
   - InfiniBand for multi-node
   - NVLink for multi-GPU same node

## Monitoring Training

### Check Job Status
```bash
squeue -u $USER
scontrol show job <job_id>
```

### View Real-time Logs
```bash
tail -f logs/slurm-<job_id>.out
```

### Cancel Job
```bash
scancel <job_id>
```

### Check GPU Usage
```bash
# On compute node
srun --jobid=<job_id> --pty nvidia-smi
```

## Example: 4-Node × 8 GPU Training (32 GPUs total)

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=6
#SBATCH --mem=400GB
#SBATCH --time=72:00:00
#SBATCH --partition=gpu

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun python train.py \
    --config config/local3.yaml \
    --distributed-backend nccl \
    --save-dir outputs
```

**Effective Batch Size**: If per-GPU batch size is 32:
- 32 (per GPU) × 32 (GPUs) = 1024 effective batch size

## Common SLURM Commands

```bash
# Submit job
sbatch script.sh

# Check queue
squeue -u $USER

# Job details
scontrol show job <job_id>

# Cancel job
scancel <job_id>

# View accounting info
sacct -j <job_id> --format=JobID,JobName,Partition,State,Elapsed,MaxRSS

# Interactive session (for debugging)
srun --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --time=1:00:00 --pty bash
```

## Testing Your Setup

Before running long training jobs, test with a small job:

1. Create a test config with:
   - Small model
   - Few training steps
   - Small dataset

2. Run single-node job first:
   ```bash
   sbatch scripts/slurm/slurm_train_single_node.sh
   ```

3. Then test multi-node:
   ```bash
   sbatch scripts/slurm/slurm_train.sh
   ```

## Additional Resources

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [SLURM Documentation](https://slurm.schedmd.com/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
