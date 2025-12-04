# Quick Reference Card

## ⚙️ First Time Setup

```bash
# 1. Copy example config and customize
cp config/train_example.yaml config/my_config.yaml

# 2. Edit config to update paths:
#    - data.datapath: Your data directory
#    - data.datafiles: Your dataset JSON files
#    - model.encoder.pretrained_audioencoder_path: Pretrained checkpoint

# 3. Test setup
scripts/test_distributed.sh 2
```

## 🚀 One-Line Commands

### Testing
```bash
scripts/test_distributed.sh 2          # Test 2 GPUs
scripts/check_env.py                   # Check environment
```

### Training

#### Single GPU
```bash
python train.py --config config/my_config.yaml
```

#### Single Node Multi-GPU
```bash
scripts/ssh/launch_singlenode.sh config/train_4gpu_example.yaml 4
```

#### Multi-Node SSH
```bash
scripts/ssh/launch_torchrun_auto.sh config/train_4gpu_example.yaml 4 node1 node2
```

#### SLURM
```bash
sbatch scripts/slurm/slurm_train.sh
```

---

## 📁 File Locations

| What | Where |
|------|-------|
| Example configs | `config/*_example.yaml` |
| Your configs | `config/*.yaml` (create from examples) |
| Launch scripts (SLURM) | `scripts/slurm/` |
| Launch scripts (SSH) | `scripts/ssh/` |
| Test utilities | `scripts/` |
| Documentation | `docs/` |
| Logs | `logs/` |
| Outputs | `outputs/` |

---

## 🔧 Configuration Quick Edit

```yaml
# config/my_config.yaml (copy from train_example.yaml)

# Data paths (REQUIRED - update these!)
data.datapath: '/path/to/your/data'
data.datafiles: ['datafiles/your_dataset.json']

# Model paths (REQUIRED - update this!)
model.encoder.pretrained_audioencoder_path: '/path/to/pretrained/encoder'

# Model options
model.encoder.audioenc_name: 'HTSAT'  # or 'Cnn14'
model.decoder.text_decoder: "HuggingFaceTB/SmolLM2-135M"

# Training
train.batch_size: 4           # Per-GPU
train.num_workers: 4          # DataLoader workers
train.num_epochs: 1000
train.learning_rate: 1e-3

# Mixed Precision (recommended)
train.mixed_precision.use_mixed_precision: True
```

---

## 📊 Monitoring

```bash
# SLURM
squeue -u $USER                # Job status
scontrol show job <jobid>      # Job details
tail -f logs/slurm-*.out       # View logs
scancel <jobid>                # Cancel job

# GPU Monitoring
nvidia-smi                     # One-time check
watch -n1 nvidia-smi          # Live monitoring

# Process Monitoring
ps aux | grep train.py         # Find processes
pkill -f "train.py"           # Kill all training
```

---

## 🐛 Quick Fixes

| Problem | Solution |
|---------|----------|
| Duplicate GPU error | Use `torchrun`, not `python` directly |
| Connection timeout | Check MASTER_ADDR, firewall, network |
| CUDA OOM | Reduce batch_size or enable mixed precision |
| Import errors | `pip install -r requirements.txt` |
| Wrong Python | `conda activate mellow` |

---

## 📦 Installation Snippets

```bash
# Full setup
conda create -n mellow python=3.10
conda activate mellow
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

# Just dependencies
pip install -r requirements.txt

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🎯 Scaling Guide

| Scenario | Command | Effective Batch Size* |
|----------|---------|----------------------|
| 1 GPU | `python train.py ...` | 4 |
| 4 GPUs | `scripts/ssh/launch_singlenode.sh ... 4` | 16 |
| 2×4 GPUs | `scripts/ssh/launch_torchrun_auto.sh ... 4 node1 node2` | 32 |
| 4×8 GPUs | `scripts/ssh/launch_torchrun_auto.sh ... 8 n1 n2 n3 n4` | 128 |

*Assuming batch_size=4 per GPU in config

---

## 🔍 Debug Checklist

Before asking for help:

- [ ] Ran `scripts/test_distributed.sh`
- [ ] Checked `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Using `torchrun` (not `python` directly for multi-GPU)
- [ ] Checked firewall/network for multi-node
- [ ] Reviewed logs in `logs/` directory
- [ ] Checked [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 📖 Documentation Map

| Document | When to Read |
|----------|--------------|
| [README.md](../README.md) | Start here! |
| [DISTRIBUTED_TRAINING_README.md](DISTRIBUTED_TRAINING_README.md) | Quick distributed overview |
| [LAUNCH_GUIDE.md](LAUNCH_GUIDE.md) | Complete launch documentation |
| [QUICKSTART_SLURM.md](QUICKSTART_SLURM.md) | SLURM TL;DR |
| [SLURM_SETUP.md](SLURM_SETUP.md) | Detailed SLURM guide |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | When things go wrong |

---

## 💡 Pro Tips

1. **Always test with 2 GPUs first**: `scripts/test_distributed.sh 2`
2. **Use tmux/screen for SSH sessions**: Prevents disconnection issues
3. **Enable mixed precision**: Free 2-3x speedup
4. **Monitor first epoch closely**: Catch configuration errors early
5. **Save checkpoints frequently**: Set `sav_per_num_epochs: 1`
6. **Check GPU utilization**: Should be 90%+ during training
7. **Only rank 0 logs**: Clean output, no duplicates

---

## 🎓 Learning Path

1. ✅ Install and verify setup
2. ✅ Run single GPU training
3. ✅ Test multi-GPU on single node
4. ✅ Try SLURM or SSH multi-node
5. ✅ Optimize hyperparameters
6. ✅ Scale to production

---

**Need more details?** Check the full [README.md](../README.md) or specific guides in [docs/](.)
