# SLURM & SSH Multi-GPU Distributed Training Setup

This repository now supports multi-GPU distributed training via both SLURM and direct SSH access.

## Quick Start

### 1. Test Your Setup
```bash
# Test with 2 GPUs (or however many you have)
scripts/test_distributed.sh 2
```

### 2. Launch Training

**On SLURM cluster:**
```bash
sbatch scripts/slurm/slurm_train_single_node.sh  # Single node
sbatch scripts/slurm/slurm_train.sh              # Multi-node
```

**With direct SSH access:**
```bash
scripts/ssh/launch_singlenode.sh config/local3.yaml 4                    # Single node, 4 GPUs
scripts/ssh/launch_torchrun_auto.sh config/local3.yaml 4 node1 node2    # Multi-node, auto
```

## Available Launch Scripts

### SLURM-Based
| Script | Purpose |
|--------|---------|
| `slurm_train_single_node.sh` | Single node testing (1 node, multiple GPUs) |
| `slurm_train.sh` | Production multi-node training |

### SSH-Based (Direct Terminal Access)
| Script | Purpose | Best For |
|--------|---------|----------|
| `launch_singlenode.sh` | Single node, multi-GPU | Testing and development |
| `launch_torchrun_auto.sh` | Multi-node automated | Easy multi-node deployment |
| `launch_torchrun_multinode.sh` | Multi-node manual | Fine control per node |
| `launch_multinode.sh` | Multi-node with torch.distributed.launch | Older PyTorch versions |
| `launch_pdsh_multinode.sh` | Multi-node with pdsh | Large clusters |

### Testing
| Script | Purpose |
|--------|---------|
| `test_distributed.sh` | Verify multi-GPU setup works |

## Detailed Documentation

- **[LAUNCH_GUIDE.md](LAUNCH_GUIDE.md)** - Complete guide with examples and troubleshooting

## What Changed?

### Modified Files
- `distributed/torch.py` - Added SLURM environment variable support
  - Detects `SLURM_LOCALID`, `SLURM_PROCID`, `SLURM_NTASKS`
  - Automatically determines master node from `SLURM_NODELIST`
  - Handles both SLURM and standard PyTorch distributed environments

### New Files
- Launch scripts (7 different options for various scenarios)
- `LAUNCH_GUIDE.md` - Comprehensive usage guide
- `test_distributed.sh` - Setup verification tool

## Requirements

- PyTorch with CUDA support
- NCCL backend (recommended) or Gloo
- For SSH-based multi-node: passwordless SSH between nodes
- For pdsh method: `pdsh` package installed

## Architecture Support

✅ Single node, single GPU  
✅ Single node, multiple GPUs  
✅ Multi-node, multiple GPUs per node  
✅ SLURM clusters  
✅ Direct SSH access  
✅ Mixed precision training  

## Environment Variables

The code automatically handles:
- SLURM variables (`SLURM_LOCALID`, `SLURM_PROCID`, `SLURM_NTASKS`, etc.)
- PyTorch DDP variables (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`)

No manual environment configuration needed when using the provided scripts!

## Examples

### Example 1: Quick Test on 4 GPUs
```bash
scripts/test_distributed.sh 4
scripts/ssh/launch_singlenode.sh config/local3.yaml 4
```

### Example 2: SLURM Multi-Node
```bash
# Edit slurm_train.sh to set your requirements
# Then submit:
sbatch scripts/slurm/slurm_train.sh

# Monitor
squeue -u $USER
tail -f logs/slurm-JOBID.out
```

### Example 3: SSH Multi-Node (3 nodes × 8 GPUs)
```bash
scripts/ssh/launch_torchrun_auto.sh config/local3.yaml 8 \
    gpu-node1 gpu-node2 gpu-node3

# Check logs
tail -f logs/node_*.log
```

## Troubleshooting

Run the test script first:
```bash
scripts/test_distributed.sh 2
```

**Common issue: "Duplicate GPU detected"**
- ✅ Always use `torchrun` or the provided launch scripts
- ❌ Don't run with `python train.py --distributed-backend nccl` directly

For detailed troubleshooting, see:
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and quick fixes
- [LAUNCH_GUIDE.md](LAUNCH_GUIDE.md) - Complete troubleshooting section

## Tips

1. **Start with single node** - Test `scripts/ssh/launch_singlenode.sh` first
2. **Use tmux/screen** - For persistent SSH sessions
3. **Monitor GPU usage** - `watch -n1 nvidia-smi` on each node
4. **Check logs** - All scripts save logs to `logs/` directory
5. **Verify connectivity** - Ensure nodes can reach each other

## Support

For issues or questions:
1. Check [LAUNCH_GUIDE.md](LAUNCH_GUIDE.md) troubleshooting section
2. Run `scripts/test_distributed.sh` to verify setup
3. Check logs in `logs/` directory

## Additional Notes

- The code handles rank and world size automatically
- Random seeds are set per-rank for reproducibility
- Gradient synchronization happens automatically via DDP
- Checkpoints save only on rank 0 to avoid conflicts
