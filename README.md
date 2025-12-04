# Mellow: a small audio language model for reasoning
[[`Paper`](https://arxiv.org/abs/2503.08540)] [[`GitHub`](https://github.com/soham97/Mellow)] [[`Checkpoint`](https://huggingface.co/soham97/Mellow)] [[`Zenodo`](https://zenodo.org/records/15036628)] [[`Demo`](https://tinyurl.com/mellowredirect)]

Mellow is a small Audio-Language Model that takes in two audios and a text prompt as input and produces free-form text as output. It is a 167M parameter model and trained on ~155 hours of audio (AudioCaps and Clotho), and achieves SoTA performance on different tasks with 50x fewer parameters. This branch contains the code to train mellow-like models. 

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Training](#training)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Scripts & Tools](#scripts--tools)
- [Configuration](#configuration)
- [Citation](#citation)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/soham97/mellow-private.git
cd mellow-private

# Create environment
conda create -n mellow python=3.10
conda activate mellow

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
# Test your multi-GPU setup
scripts/test_distributed.sh 2  # Test with 2 GPUs
```

### 3. Train

**Single GPU:**
```bash
python train.py --config config/local3.yaml
```

**Multi-GPU (4 GPUs on single node):**
```bash
scripts/ssh/launch_singlenode.sh config/local3.yaml 4
```

**SLURM Cluster:**
```bash
sbatch scripts/slurm/slurm_train.sh
```

**SSH Multi-Node:**
```bash
scripts/ssh/launch_torchrun_auto.sh config/local3.yaml 4 node1 node2
```

---

## 📦 Installation

### Prerequisites

- **Python**: 3.10 or higher
- **CUDA**: 11.6+ (for GPU training)
- **PyTorch**: 1.12.1 or higher
- **Hardware**: Multi-GPU support requires NCCL

### Detailed Setup

```bash
# 1. Create conda environment
conda create -n mellow python=3.10
conda activate mellow

# 2. Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 5. Test distributed setup (optional)
scripts/test_distributed.sh 2
```

### ⚠️ Important: Update Conda Environment in Launch Scripts

**Before using SLURM or SSH launch scripts**, update the conda environment name:

All launch scripts in `scripts/slurm/` and `scripts/ssh/` default to activating `qa_gen_3.1`. 

**To use your own environment:**

Open any launch script and change this line:
```bash
conda activate qa_gen_3.1  # Change to your environment name
```

For example, if you created an environment called `mellow`:
```bash
conda activate mellow
```

Scripts to update:
- All files in `scripts/slurm/` (4 scripts)
- All files in `scripts/ssh/` (5 scripts)

Or use a quick find-replace:
```bash
# Replace in all SLURM scripts
sed -i 's/qa_gen_3.1/mellow/g' scripts/slurm/*.sh

# Replace in all SSH scripts
sed -i 's/qa_gen_3.1/mellow/g' scripts/ssh/*.sh
```

---

## 🎯 Training

### Configuration Setup

Before training, create your configuration file from the examples:

```bash
# Copy example config and modify for your setup
cp config/train_example.yaml config/my_training.yaml

# Edit the config file to update:
# - datapath: Path to your data directory
# - datafiles: List of JSON files with your dataset
# - pretrained_audioencoder_path: Path to pretrained audio encoder
```

**Important paths to update in your config:**
- `data.datapath`: Root directory containing your audio files
- `data.datafiles`: JSON files with dataset metadata (see `datafiles/` examples)
- `model.encoder.pretrained_audioencoder_path`: Path to pretrained HTSAT checkpoint

### Single GPU

```bash
python train.py \
    --config config/my_training.yaml \
    --save-dir outputs
```

### Multi-GPU (Single Node)

**Using convenience script (recommended):**
```bash
scripts/ssh/launch_singlenode.sh config/train_4gpu_example.yaml 4
```

**Using torchrun directly:**
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    train.py \
    --config config/train_4gpu.yaml \
    --distributed-backend nccl
```

### Multi-Node Training

#### SLURM Cluster

**Quick test (single node):**
```bash
sbatch scripts/slurm/slurm_train_single_node.sh
```

**Full run (multi-node):**
```bash
# Edit scripts/slurm/slurm_train.sh to configure:
# - Number of nodes
# - GPUs per node
# - Time limit, memory, partition

sbatch scripts/slurm/slurm_train.sh

# Monitor
squeue -u $USER
tail -f logs/slurm-<job_id>.out
```

#### SSH-Based Multi-Node

**Automated (easiest):**
```bash
scripts/ssh/launch_torchrun_auto.sh config/local3.yaml 4 node1 node2 node3
```

**Manual control:**
```bash
# On each node separately
NODE_RANK=0 scripts/ssh/launch_torchrun_multinode.sh config/local3.yaml 3 4 node1  # node1
NODE_RANK=1 scripts/ssh/launch_torchrun_multinode.sh config/local3.yaml 3 4 node1  # node2
NODE_RANK=2 scripts/ssh/launch_torchrun_multinode.sh config/local3.yaml 3 4 node1  # node3
```

### Evaluation

```bash
# First, create your evaluation config
cp config/eval_example.yaml config/my_eval.yaml

# Edit config to set:
# - datapath and datafiles for evaluation data
# - resume_checkpoint: path to trained model

# Run evaluation
python train.py \
    --config config/my_eval.yaml \
    --mode evaluate_checkpoint \
    --checkpoint_path outputs/model-epo-10.ckpt
```

---

## 📁 Project Structure

```
mellow-private/
├── README.md                   # This file
├── train.py                    # Main training script
├── requirements.txt            # Python dependencies
│
├── config/                     # Configuration files
│   ├── train_example.yaml     # Example single-GPU config
│   ├── train_4gpu_example.yaml # Example 4-GPU config
│   ├── eval_example.yaml      # Example evaluation config
│   ├── local3.yaml            # (your custom configs here)
│   └── ...
│
├── models/                     # Model architectures
│   ├── mellow.py              # Main model
│   ├── audio.py               # Audio encoders (HTSAT, CNN14)
│   ├── decoder.py             # Text decoders
│   └── generate.py            # Generation utilities
│
├── data/                       # Data loading
│   ├── audiotext_dataset.py   # Training dataset
│   ├── audiotext_eval_dataset.py  # Evaluation dataset
│   └── sampler.py             # Distributed sampler
│
├── training/                   # Training framework
│   ├── trainer.py             # Main trainer class
│   └── log.py                 # Logging utilities
│
├── distributed/                # Distributed training
│   ├── torch.py               # PyTorch DDP wrapper
│   └── __init__.py
│
├── metrics/                    # Evaluation metrics
│   ├── aqa.py                 # Audio quality assessment
│   └── capmetrics.py          # Caption metrics
│
├── utils/                      # Utilities
│   ├── launch_utils.py        # Launch helpers
│   └── utils.py               # General utilities
│
├── scripts/                    # Launch scripts & tools
│   ├── test_distributed.sh    # Test multi-GPU setup
│   ├── check_env.py           # Check environment variables
│   ├── verify_slurm_setup.py  # Verify SLURM setup
│   │
│   ├── slurm/                 # SLURM launch scripts
│   │   ├── slurm_train.sh            # Multi-node run
│   │   ├── slurm_train_single_node.sh # Single node testing
│   │   ├── slurm_train_torchrun.sh   # Alternative with torchrun
│   │   └── slurm_debug.sh            # Debug environment
│   │
│   └── ssh/                   # SSH-based launch scripts
│       ├── launch_singlenode.sh          # Single node, multi-GPU
│       ├── launch_torchrun_auto.sh       # Auto multi-node
│       ├── launch_torchrun_multinode.sh  # Manual multi-node
│       ├── launch_multinode.sh           # Legacy launcher
│       └── launch_pdsh_multinode.sh      # Using pdsh
│
└── docs/                       # Documentation
    ├── DISTRIBUTED_TRAINING_README.md  # Quick distributed guide
    ├── LAUNCH_GUIDE.md                 # Comprehensive launch guide
    ├── QUICKSTART_SLURM.md            # SLURM quick reference
    ├── SLURM_SETUP.md                 # Detailed SLURM guide
    └── TROUBLESHOOTING.md             # Common issues & solutions
```

---

## 📚 Documentation

### Core Guides

- **[Quick Start](docs/DISTRIBUTED_TRAINING_README.md)** - Get started with distributed training in 5 minutes
- **[Launch Guide](docs/LAUNCH_GUIDE.md)** - Comprehensive guide for all launch scenarios
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and quick fixes

### SLURM-Specific

- **[SLURM Quick Start](docs/QUICKSTART_SLURM.md)** - TL;DR for SLURM users
- **[SLURM Setup Guide](docs/SLURM_SETUP.md)** - Detailed SLURM configuration

---

## 🛠️ Scripts & Tools

### Testing & Verification

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/test_distributed.sh` | Test multi-GPU setup | `scripts/test_distributed.sh 4` |
| `scripts/check_env.py` | Check environment variables | `torchrun ... check_env.py` |
| `scripts/verify_slurm_setup.py` | Verify SLURM configuration | `python scripts/verify_slurm_setup.py` |

### SLURM Launch Scripts

| Script | Purpose | Nodes | GPUs |
|--------|---------|-------|------|
| `scripts/slurm/slurm_train.sh` |  Multi-node run | 2+ | 4+ per node |
| `scripts/slurm/slurm_train_single_node.sh` | Single node testing | 1 | 4 |
| `scripts/slurm/slurm_train_torchrun.sh` | Alternative launcher | 2+ | 4+ per node |
| `scripts/slurm/slurm_debug.sh` | Debug environment | 1 | 2 |

### SSH Launch Scripts

| Script | Purpose | Best For |
|--------|---------|----------|
| `scripts/ssh/launch_singlenode.sh` | Single node, multi-GPU | Development & testing |
| `scripts/ssh/launch_torchrun_auto.sh` | Automated multi-node | Easy deployment |
| `scripts/ssh/launch_torchrun_multinode.sh` | Manual multi-node | Fine control |
| `scripts/ssh/launch_multinode.sh` | Legacy launcher | Older PyTorch |
| `scripts/ssh/launch_pdsh_multinode.sh` | Parallel shell | Large clusters |

---

## ⚙️ Configuration

### Model Configuration

Edit YAML files in `config/` directory:

```yaml
model:
  encoder:
    audioenc_name: 'HTSAT'  # or 'Cnn14'
    out_emb: 768
    d_proj: 576
    use_pretrained_audioencoder: True
    freeze_audio_encoder_weights: True
    
  decoder:
    text_decoder: "HuggingFaceTB/SmolLM2-135M"
    prefix_length: 40
    freeze_gpt_weights: False

train:
  batch_size: 4                # Per-GPU batch size
  num_epochs: 1000
  learning_rate: 1e-3
  num_workers: 4               # DataLoader workers
  mixed_precision:
    use_mixed_precision: True
    mixed_precision_dtype: "float16"
```

### Audio Encoders

- **HTSAT**: HTS-Audio Transformer (recommended for best quality)
- **Cnn14**: CNN-based audio encoder

### Text Decoders

- GPT-2 (all sizes: small, medium, large, xl)
- SmolLM2 (135M, 360M, 1.7B)
- Any HuggingFace causal language model

### Command-Line Overrides

```bash
python train.py \
    --config config/local3.yaml \
    --train.batch_size 8 \
    --train.learning_rate 5e-4 \
    --model.decoder.text_decoder "gpt2-medium"
```

---

## 🎓 Some Features Explained

### Distributed Training

- ✅ **Automatic rank detection** from SLURM, torchrun, or manual env vars
- ✅ **NCCL backend** for efficient GPU communication
- ✅ **Gradient synchronization** handled automatically by DDP
- ✅ **Clean logging** - only rank 0 logs to avoid spam
- ✅ **Checkpoint management** - automatic saving/loading with rank coordination

### Performance Optimizations

- **Mixed Precision Training**: 2-3x speedup with FP16/BF16
- **Gradient Accumulation**: Effective large batch sizes
- **Efficient Data Loading**: Multi-worker data loading with proper seeding
- **SyncBatchNorm**: Synchronized batch normalization across GPUs
- **No find_unused_parameters**: Optimized DDP without unnecessary overhead

### Logging & Monitoring

- **Rank-aware logging**: Only rank 0 produces output
- **Worker log management**: DataLoader workers properly silenced
- **Progress tracking**: TensorBoard compatible logging
- **Error handling**: Comprehensive error messages with context

---

## 🔍 Quick Reference

### Environment Variables

Set by launcher (don't set manually):
- `RANK` - Global rank (0 to world_size-1)
- `WORLD_SIZE` - Total number of processes
- `LOCAL_RANK` - Local rank on node (0 to GPUs-1)
- `MASTER_ADDR` - Master node address
- `MASTER_PORT` - Communication port

### Common Commands

```bash
# Test setup
scripts/test_distributed.sh 2

# Single node, 4 GPUs
scripts/ssh/launch_singlenode.sh config/local3.yaml 4

# Multi-node SSH (3 nodes × 8 GPUs = 24 GPUs)
scripts/ssh/launch_torchrun_auto.sh config/local3.yaml 8 node1 node2 node3

# SLURM
sbatch scripts/slurm/slurm_train.sh

# Check SLURM job
squeue -u $USER
tail -f logs/slurm-<job_id>.out

# Debug environment
torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/check_env.py
```

---

## ⚠️ Troubleshooting

**Common Issues:**

1. **"Duplicate GPU detected"** - Use `torchrun`, not `python` directly
2. **"I/O operation on closed file"** - Already fixed in code (DataLoader worker logging)
3. **Connection timeout** - Check firewall, MASTER_ADDR, network connectivity
4. **CUDA OOM** - Reduce batch size or enable mixed precision

**Full troubleshooting guide:** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## 🎯 Performance Tips

1. **Scale batch size with GPUs**: 32 per GPU × 8 GPUs = 256 effective batch size
2. **Enable mixed precision**: 2-3x speedup with minimal accuracy loss
3. **Optimize data loading**: Set `num_workers` to `cpus_per_task - 1`
4. **Use high-speed interconnects**: InfiniBand for multi-node, NVLink for multi-GPU
5. **Monitor GPU utilization**: `watch -n1 nvidia-smi`

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{
deshmukh2025mellow,
title={Mellow: a small audio language model for reasoning},
author={Soham Deshmukh and Satvik Dixit and Rita Singh and Bhiksha Raj},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=um4aiicz3L}
}
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `scripts/test_distributed.sh`
5. Submit a pull request

---

## 📧 Contact

For questions, issues, or collaboration:

- **GitHub Issues**: [Create an issue](https://github.com/soham97/mellow/issues)

---

**Note**: This is a research codebase and not meant for production. For production use, additional testing and optimization may be required.
