# Configuration Files

This directory contains configuration files for training and evaluation.

## Getting Started

### Step 1: Choose a Template

Copy one of the example config files based on your use case:

```bash
# For single GPU training
cp train_example.yaml my_training.yaml

# For multi-GPU (4 GPUs) training
cp train_4gpu_example.yaml my_training.yaml

# For evaluation
cp eval_example.yaml my_eval.yaml
```

### Step 2: Update Required Paths

Edit your config file to update the following **required** paths:

```yaml
data:
    # Path to your data directory
    datapath: '/path/to/your/data'
    
    # List of JSON files containing dataset metadata
    datafiles:
        - 'datafiles/your_dataset.json'

model:
    encoder:
        # Path to pretrained audio encoder checkpoint
        pretrained_audioencoder_path: '/path/to/pretrained/encoder'
    
    # For evaluation only: path to trained model
    resume_checkpoint: "/path/to/trained/model.ckpt"
```

### Step 3: Adjust Training Parameters (Optional)

```yaml
train:
    batch_size: 4              # Per-GPU batch size
    num_workers: 4             # Data loading workers per GPU
    num_epochs: 10             # Number of training epochs
    sync_batchnorm: True       # Use for multi-GPU training
    learning_rate: 1e-3
```

## Configuration Structure

### Data Section
- `datapath`: Root directory containing audio files
- `datafiles`: List of JSON files with metadata (filepath, caption, etc.)
- `sampling_rate`: Audio sampling rate (default: 32000)
- `segment_seconds`: Audio segment length in seconds
- `tokenizer_type`: HuggingFace tokenizer model
- `op_text_len`: Output text length
- `ip_text_len`: Input text length

### Model Section

#### Encoder
- `audioenc_name`: Audio encoder type (`'HTSAT'` or `'Cnn14'`)
- `pretrained_audioencoder_path`: Path to pretrained encoder checkpoint
- `freeze_audio_encoder_weights`: Whether to freeze encoder during training

#### Decoder
- `text_decoder`: HuggingFace text decoder model
- `prefix_length`: Length of audio prefix
- `freeze_gpt_weights`: Whether to freeze decoder during training

### Train Section
- `optimizer`: Optimizer configuration (type, learning rate, scheduler)
- `batch_size`: Per-GPU batch size
- `num_workers`: DataLoader workers per GPU
- `num_epochs`: Number of training epochs
- `sync_batchnorm`: Synchronize batch normalization across GPUs
- `mixed_precision`: Mixed precision training settings

## Example Configs

- **`train_example.yaml`**: Single GPU training template
- **`train_4gpu_example.yaml`**: Multi-GPU (4 GPUs) training template
- **`eval_example.yaml`**: Evaluation template

## Data File Format

Your JSON datafiles should contain entries like:

```json
[
    {
        "filepath1": "path/to/audio1.wav",
        "filepath2": "",
        "input": "Question or prompt text",
        "answer": "Expected answer",
        "caption1": "Audio description",
        "caption2": ""
    }
]
```

## Notes

- **Do not commit** your custom config files with personal paths to git
- The `.gitignore` is configured to track only `*_example.yaml` files
- For multi-GPU training, set `sync_batchnorm: True` and adjust `num_workers`
- For evaluation, set `mode: 'evaluate_checkpoint'` and provide `resume_checkpoint`
