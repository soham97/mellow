# Quick Start: SLURM Multi-GPU Training

## TL;DR - Get Started in 3 Steps

### 1. Test Your Setup (Debug Mode - 2 GPUs, 1 hour)
```bash
sbatch scripts/slurm/slurm_debug.sh
```
Check logs: `tail -f logs/debug-<job_id>.out`

### 2. Single Node Training (4 GPUs)
```bash
# Edit slurm_train_single_node.sh if needed (adjust partition, time, memory)
sbatch scripts/slurm/slurm_train_single_node.sh

# Monitor
squeue -u $USER
tail -f logs/slurm-<job_id>.out
```

### 3. Multi-Node Training (2 nodes × 4 GPUs = 8 GPUs)
```bash
# Edit slurm_train.sh if needed
sbatch scripts/slurm/slurm_train.sh

# Monitor
squeue -u $USER
tail -f logs/slurm-<job_id>.out
```

## What Changed in the Code?

The `distributed/torch.py` file now automatically detects SLURM environment variables:

- **SLURM_PROCID** → Global rank
- **SLURM_LOCALID** → Local rank (GPU ID on node)
- **SLURM_NTASKS** → World size (total GPUs)
- **SLURM_NODELIST** → Node names (used to set MASTER_ADDR)

No code changes needed in your training script!

## Which Script Should I Use?

| Script | Use Case | Nodes | GPUs | Notes |
|--------|----------|-------|------|-------|
| `slurm_debug.sh` | Quick test | 1 | 2 | 1 hour max, debug output |
| `slurm_train_single_node.sh` | Single node | 1 | 4 | Simple, good for testing |
| `slurm_train.sh` | Multi-node | 2+ | 4+ per node | Full scale training |
| `slurm_train_torchrun.sh` | Alternative launcher | 2+ | 4+ per node | Uses torchrun (PyTorch 1.10+) |

## Customize for Your Cluster

Edit the `#SBATCH` lines in the scripts:

```bash
#SBATCH --partition=gpu              # Your cluster's GPU partition name
#SBATCH --time=48:00:00              # Adjust time limit
#SBATCH --mem=200GB                  # Adjust memory
#SBATCH --gres=gpu:4                 # Number of GPUs (gpu:2, gpu:4, gpu:8, etc.)
```

## Common Adjustments

### Change Number of Nodes/GPUs

**For 4 nodes × 8 GPUs (32 GPUs total):**
```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
```

### Change Config File
```bash
srun python train.py \
    --config config/YOUR_CONFIG.yaml \
    --distributed-backend nccl \
    --save-dir outputs
```

### Enable Conda Environment

Uncomment in the script:
```bash
# Activate conda environment
source activate your_env_name
```

Or add after `#!/bin/bash`:
```bash
source ~/.bashrc
conda activate mellow
```

## Check Job Status

```bash
# View queue
squeue -u $USER

# Job details
scontrol show job <job_id>

# Cancel job
scancel <job_id>

# View logs
tail -f logs/slurm-<job_id>.out
tail -f logs/slurm-<job_id>.err
```

## Troubleshooting

### Job doesn't start?
- Check queue: `squeue -u $USER`
- Check partition exists: `sinfo`
- Check your limits: `sacctmgr show user $USER`

### "Connection timeout" error?
- Ensure nodes can communicate
- Check MASTER_ADDR is set correctly
- Try different MASTER_PORT (change 29500 to 29501, etc.)

### "NCCL error"?
- Add to script: `export NCCL_DEBUG=INFO`
- Check CUDA/NCCL modules loaded correctly
- Verify GPU visibility: `echo $CUDA_VISIBLE_DEVICES`

### "Out of memory"?
- Reduce batch size in config file
- Increase `#SBATCH --mem=`
- Use fewer GPUs per node

## Performance Tips

1. **Scale batch size with GPUs**: If you use 32 per GPU, with 8 GPUs → 256 effective batch size
2. **Use local NVMe if available**: Copy data to node's local storage first
3. **Set num_workers**: Usually `cpus_per_task - 1`
4. **Use NCCL for GPU training**: Already configured with `--distributed-backend nccl`

## Next Steps

1. Start with `slurm_debug.sh` to verify everything works
2. Move to `slurm_train_single_node.sh` for real training on 1 node
3. Scale to `slurm_train.sh` for multi-node when needed

## Need Help?

Check the detailed guide: `SLURM_SETUP.md`
