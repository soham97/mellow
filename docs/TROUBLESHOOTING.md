# Common Issues Quick Reference

## Error: "Duplicate GPU detected"

**Full error:**
```
Duplicate GPU detected : rank X and rank Y both on CUDA device XXXXX
ncclInvalidUsage: This usually reflects invalid usage of NCCL library
```

**Solution:**
✅ **Use `torchrun` to launch, not `python` directly**

```bash
# ✅ CORRECT
torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py --config config.yaml --distributed-backend nccl

# ❌ WRONG - will cause duplicate GPU error
python train.py --config config.yaml --distributed-backend nccl

# ✅ Or use the provided scripts
scripts/ssh/launch_singlenode.sh config.yaml 4
```

**Why:** `torchrun` properly sets up environment variables (LOCAL_RANK, RANK, WORLD_SIZE) for each process. Running with `python` directly doesn't create separate processes with proper rank assignment.

---

## Error: All processes show same local_rank (hang at initialization)

**Symptoms:**
- Logs show all processes with `local_rank = 0`
- Training hangs after model initialization
- All processes trying to use same GPU

**Debug:**
Check what environment variables are set:
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 check_env.py
```

This will show you what `LOCAL_RANK`, `RANK`, etc. are set for each process.

**Solution:**
The code now automatically detects local rank from:
1. `LOCAL_RANK` (set by torchrun)
2. `SLURM_LOCALID` (set by SLURM)
3. Computed from `RANK % SLURM_NTASKS_PER_NODE`

If none are available, you'll get a clear error message with available variables.

---

## Error: "Unable to determine local rank"

**Solution:**
Ensure you're using one of these launchers:
- `torchrun` (sets LOCAL_RANK automatically)
- SLURM with `srun` (sets SLURM_LOCALID)
- Our launch scripts (handle everything)

---

## Error: "ValueError: I/O operation on closed file" in DataLoader workers

**Symptoms:**
- Error appears in worker processes during data loading
- Training may continue but with logging errors
- Stack trace shows error in `logging/__init__.py`

**Cause:** 
DataLoader workers (multiprocessing) trying to write to closed log files.

**Solution:**
✅ **Already fixed in the code**
- Worker logging is now more robust with fallback
- Warnings from transformers tokenizers are suppressed in workers
- Only worker 0 logs to reduce noise

**Additional steps if still seeing issues:**
1. Reduce number of workers:
   ```yaml
   train:
     num_workers: 2  # Reduce from 4
   ```

2. Disable persistent workers:
   ```yaml
   train:
     persistent_data_workers: False
   ```

---

## Warning: "find_unused_parameters=True" in DDP

**Warning:**
```
[W reducer.cpp:1346] Warning: find_unused_parameters=True was specified in DDP constructor, 
but did not find any unused parameters in the forward pass.
```

**Solution:**
✅ **Already fixed** - Changed to `find_unused_parameters=False` for better performance.

If you have a model with conditional paths (different parameters used in different forward passes), 
you may need to set it back to `True`.

---

## Warning: Tokenizer deprecation warnings

**Warnings:**
- "Truncation was not explicitly activated"
- "The `pad_to_max_length` argument is deprecated"

**Solution:**
✅ **Already fixed** - Updated tokenizer calls to use:
- `truncation=True`
- `padding='max_length'` instead of `pad_to_max_length=True`

These warnings are also suppressed in DataLoader workers.

---

## Error: Connection timeout or "Address already in use"

**Solutions:**

1. **Different MASTER_PORT:**
   ```bash
   export MASTER_PORT=29501
   torchrun --standalone --nproc_per_node=4 train.py ...
   ```

2. **Check port availability:**
   ```bash
   netstat -tuln | grep 29500
   ```

3. **Kill zombie processes:**
   ```bash
   pkill -f "train.py"
   ```

---

## Error: "CUDA out of memory"

**Solutions:**

1. **Reduce batch size** in config:
   ```yaml
   train:
     batch_size: 2  # Reduce from 4
   ```

2. **Enable mixed precision:**
   ```yaml
   train:
     mixed_precision:
       use_mixed_precision: True
       mixed_precision_dtype: "float16"
   ```

3. **Use gradient accumulation** (if supported in your training loop)

4. **Check GPU memory:**
   ```bash
   nvidia-smi
   ```

---

## How to Launch Correctly

### ✅ Single Node Multi-GPU

```bash
# Option 1: Use script
scripts/ssh/launch_singlenode.sh config/local3.yaml 4

# Option 2: Use torchrun directly
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    train.py --config config/local3.yaml --distributed-backend nccl
```

### ✅ Multi-Node (SSH)

```bash
# Automated launch on multiple nodes
scripts/ssh/launch_torchrun_auto.sh config/local3.yaml 4 node1 node2
```

### ✅ Multi-Node (SLURM)

```bash
# Submit job
sbatch scripts/slurm/slurm_train.sh
```

---

## Pre-flight Checklist

Before launching distributed training:

1. ✅ **Check GPUs are visible:**
   ```bash
   nvidia-smi
   ```

2. ✅ **Test with small number of GPUs first:**
   ```bash
   scripts/test_distributed.sh 2
   ```

3. ✅ **Verify PyTorch can see GPUs:**
   ```bash
   python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
   ```

4. ✅ **Check NCCL is available:**
   ```bash
   python -c "import torch.distributed as dist; print(f'NCCL: {dist.is_nccl_available()}')"
   ```

5. ✅ **Use the test script:**
   ```bash
   scripts/test_distributed.sh 4
   ```

---

## Quick Debugging

### Check environment variables set by launcher:
```bash
# Check what torchrun sets
torchrun --standalone --nnodes=1 --nproc_per_node=4 check_env.py

# Check in SLURM
srun --nodes=1 --ntasks-per-node=4 --gres=gpu:4 python check_env.py
```

### Enable verbose NCCL logging:
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
torchrun --standalone --nproc_per_node=4 train.py ...
```

### Check what's running:
```bash
ps aux | grep train.py
nvidia-smi
```

### Kill all training processes:
```bash
pkill -f "train.py"
# or
killall -9 python
```

### Check logs:
```bash
# For SLURM
tail -f logs/slurm-*.out

# For SSH launches
tail -f logs/node_*.log
```

---

## Logging Configuration

By default, **only rank 0 (main process) logs** to keep output clean:

- **Main process (rank 0)**: Full logging at INFO level
- **Other processes (rank > 0)**: Only ERROR level (critical errors only)
- **DataLoader workers**: Only CRITICAL level (essentially silent)

This prevents duplicate log messages from all processes and makes output much more readable.

**If you need to debug a specific rank**, temporarily modify `train.py`:
```python
# Original (only rank 0 logs):
if distributed_ctx.rank() > 0:
    logging.getLogger().setLevel(logging.ERROR)
    
# To debug specific rank (e.g., rank 2):
if distributed_ctx.rank() not in [0, 2]:  # Log from rank 0 and 2
    logging.getLogger().setLevel(logging.ERROR)
```

---

## Remember

- 🚫 **Never** use `CUDA_VISIBLE_DEVICES` when using torchrun
- ✅ **Always** use torchrun/srun for multi-GPU training
- ✅ **Always** specify `--distributed-backend nccl` for GPU training
- ✅ **Always** test with 2 GPUs before scaling up
- ✅ **Always** use the provided launch scripts if unsure

---

## Still Having Issues?

1. Check the full guide: [LAUNCH_GUIDE.md](LAUNCH_GUIDE.md)
2. Run the test script: `scripts/test_distributed.sh 2`
3. Enable debug logging: `export NCCL_DEBUG=INFO`
4. Check all logs in `logs/` directory
