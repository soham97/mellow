# Multi-GPU Distributed Training Launch Guide

This guide covers different ways to launch multi-GPU distributed training based on your setup.

## Prerequisites

Before launching distributed training, make sure you have:

1. **Created your config file** from the examples:
   ```bash
   cp config/train_example.yaml config/my_training.yaml
   # or for 4 GPUs:
   cp config/train_4gpu_example.yaml config/my_training.yaml
   ```

2. **Updated required paths** in your config:
   - `data.datapath`: Path to your data directory
   - `data.datafiles`: List of JSON files with dataset metadata
   - `model.encoder.pretrained_audioencoder_path`: Path to pretrained audio encoder

3. **Tested your setup**:
   ```bash
   scripts/test_distributed.sh 2
   ```

> **Note:** All examples below use `config/local3.yaml` as a placeholder. Replace it with your actual config file (e.g., `config/my_training.yaml`).

## Table of Contents
1. [SLURM-based Launch](#slurm-based-launch)
2. [Direct SSH Access Launch](#direct-ssh-access-launch)
3. [Environment Variables](#environment-variables)
4. [Troubleshooting](#troubleshooting)

---

## SLURM-based Launch

### Single Node (Testing)
```bash
sbatch scripts/slurm/slurm_train_single_node.sh
```

### Multi-Node Production
```bash
# Edit slurm_train.sh to set:
# - Number of nodes
# - GPUs per node
# - Time limit
# - Your config file path

sbatch scripts/slurm/slurm_train.sh
```

### Monitor SLURM Job
```bash
# Check job status
squeue -u $USER

# View output
tail -f logs/slurm-JOBID.out

# Cancel job
scancel JOBID
```

---

## Direct SSH Access Launch

### Method 1: Single Node (Simplest)
Best for: Testing on a single multi-GPU machine

```bash
# Using torchrun (recommended)
scripts/ssh/launch_singlenode.sh config/local3.yaml 4

# Or manually with torchrun
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    train.py --config config/local3.yaml --distributed-backend nccl

# IMPORTANT: Always use torchrun for multi-GPU training
# Don't run with: python train.py --distributed-backend nccl
# That will cause "Duplicate GPU detected" errors
```

### Method 2: Multi-Node with torchrun (Recommended)
Best for: Clean, modern PyTorch distributed training

**Automated (easiest):**
```bash
# Launches on all nodes automatically via SSH
scripts/ssh/launch_torchrun_auto.sh config/local3.yaml 4 node1 node2 node3
```

**Manual (run on each node separately):**
```bash
# On node 0 (master)
NODE_RANK=0 scripts/ssh/launch_torchrun_multinode.sh config/local3.yaml 2 4 node1

# On node 1
NODE_RANK=1 scripts/ssh/launch_torchrun_multinode.sh config/local3.yaml 2 4 node1
```

### Method 3: Multi-Node with torch.distributed.launch
Best for: Compatibility with older PyTorch versions

```bash
scripts/ssh/launch_multinode.sh config/local3.yaml 2 4 "node1,node2"
```

### Method 4: Using pdsh (Parallel Distributed Shell)
Best for: Large clusters with many nodes

```bash
# Requires pdsh installed
scripts/ssh/launch_pdsh_multinode.sh config/local3.yaml 4 "node[1-4]"

# Or with explicit list
scripts/ssh/launch_pdsh_multinode.sh config/local3.yaml 4 "node1,node2,node3,node4"
```

---

## Environment Variables

### Required for Multi-Node Training

```bash
export MASTER_ADDR=node1        # Master node hostname/IP
export MASTER_PORT=29500        # Communication port
export WORLD_SIZE=8             # Total number of processes
export RANK=0                   # Global rank (0 to WORLD_SIZE-1)
export LOCAL_RANK=0             # Local rank on this node (0 to GPUS-1)
```

### SLURM Auto-sets These
```bash
SLURM_JOB_ID                    # Job ID
SLURM_NTASKS                    # Total tasks (= WORLD_SIZE)
SLURM_PROCID                    # Process ID (= RANK)
SLURM_LOCALID                   # Local ID (= LOCAL_RANK)
SLURM_NODELIST                  # List of nodes
SLURM_GPUS_PER_NODE            # GPUs per node
```

---

## Configuration

### For Multi-Node Training
Edit your config file (e.g., `config/local3.yaml`):

```yaml
train:
  num_nodes: 2                  # Number of nodes
  batch_size: 64                # Per-GPU batch size
  mixed_precision:
    use_mixed_precision: true
```

---

## Examples

### Example 1: 4 GPUs on Single Node
```bash
scripts/ssh/launch_singlenode.sh config/local3.yaml 4
```

### Example 2: 2 Nodes × 4 GPUs = 8 GPUs Total
```bash
# Automated
scripts/ssh/launch_torchrun_auto.sh config/local3.yaml 4 gpu-node1 gpu-node2

# Or with SLURM
sbatch scripts/slurm/slurm_train.sh  # (edit to set nodes=2, ntasks-per-node=4)
```

### Example 3: 4 Nodes × 8 GPUs = 32 GPUs Total
```bash
scripts/ssh/launch_torchrun_auto.sh config/local3.yaml 8 \
    node1 node2 node3 node4
```

---

## Troubleshooting

### Issue: "Duplicate GPU detected" 
**Error:** `Duplicate GPU detected : rank X and rank Y both on CUDA device XXXXX`

**Cause:** All processes trying to use the same GPU

**Solutions:**
1. **Don't set CUDA_VISIBLE_DEVICES manually** when using torchrun - it handles device assignment automatically
2. Verify you're using torchrun (not python directly):
   ```bash
   # Correct
   torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py ...
   
   # Incorrect (don't do this with DDP)
   python train.py --distributed-backend nccl ...
   ```
3. Check that GPUs are available:
   ```bash
   nvidia-smi
   ```

### Issue: "Unable to determine local rank"
**Solution:** Set LOCAL_RANK environment variable
```bash
export LOCAL_RANK=0
```

### Issue: "Connection timeout" or "NCCL error"
**Solutions:**
1. Check network connectivity between nodes:
   ```bash
   ping node2
   ```

2. Verify MASTER_ADDR is reachable from all nodes

3. Check firewall allows port 29500 (or your MASTER_PORT)

4. Try using a different backend:
   ```bash
   --distributed-backend gloo  # Instead of nccl
   ```

### Issue: "CUDA out of memory"
**Solutions:**
1. Reduce batch size in config
2. Enable gradient checkpointing
3. Use mixed precision training

### Issue: Processes hang at initialization
**Solutions:**
1. Check all nodes have same environment:
   ```bash
   pdsh -w node[1-4] "which python"
   pdsh -w node[1-4] "python --version"
   ```

2. Verify WORLD_SIZE matches actual number of processes

3. Check logs on all nodes for errors

### Issue: Different random seeds across GPUs
**Solution:** This is handled automatically by the code based on rank

---

## Verification

Before launching, verify your setup:

```bash
# Check CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Check distributed backend
python -c "import torch.distributed as dist; print(f'NCCL available: {dist.is_nccl_available()}')"

# Test node connectivity (from master)
for node in node1 node2 node3; do
    ssh $node "hostname && nvidia-smi -L"
done
```

---

## Tips

1. **Start small:** Test with single node before scaling to multi-node
2. **Use tmux/screen:** For long-running SSH sessions
3. **Monitor GPUs:** Use `nvidia-smi` or `watch -n1 nvidia-smi`
4. **Save logs:** All scripts redirect output to `logs/` directory
5. **Gradient accumulation:** Increase effective batch size without OOM
6. **Checkpoint often:** Multi-node failures are more common

---

## Quick Reference

| Scenario | Command |
|----------|---------|
| Single node, 4 GPUs | `scripts/ssh/launch_singlenode.sh config/local3.yaml 4` |
| Multi-node, automated | `scripts/ssh/launch_torchrun_auto.sh config/local3.yaml 4 node1 node2` |
| SLURM single node | `sbatch scripts/slurm/slurm_train_single_node.sh` |
| SLURM multi-node | `sbatch scripts/slurm/slurm_train.sh` |
| With pdsh | `scripts/ssh/launch_pdsh_multinode.sh config/local3.yaml 4 "node[1-4]"` |

---

## Additional Resources

- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
