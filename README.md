# PIRE-DIRT

A modular PyTorch implementation for AI-generated image detection using:

- a locally stored **DINOv3 ViT-L/16** backbone;
- **LoRA** adapters inserted into attention QKV projections;
- a global classification branch based on the class token;
- **PIRE**: Prior Information Region Enhancement for complementary style, frequency, and noise-statistical enhancement;
- **DIRT**: Dynamic Important Region Token filtering for regional prediction;
- distributed training and evaluation with PyTorch DDP.

The repository uses a standard `src/` Python layout and separates datasets, models, training, evaluation, configuration, and utility code.

## Project structure

```text
PIRE-DIRT/
├── configs/
│   └── default.yaml
├── docs/
│   ├── PROJECT_STRUCTURE.md
│   └── README.md
├── scripts/
│   ├── train_ddp.sh
│   └── test_ddp.sh
├── src/
│   └── pire_dirt/
│       ├── data/
│       ├── engine/
│       ├── models/
│       │   ├── detector.py
│       │   ├── dirt.py
│       │   ├── lora.py
│       │   └── pire.py
│       └── utils/
├── tests/
├── train.py
├── test.py
├── requirements.txt
└── pyproject.toml
```

## Dataset layout

Each split contains one folder per domain or generator. Every domain contains real and fake image folders:

```text
AIGIBench/
├── train/
│   ├── progan/
│   │   ├── 0_real/
│   │   └── 1_fake/
│   └── sdv1.4/
│       ├── 0_real/
│       └── 1_fake/
├── val/
│   └── ...
└── test/
    ├── DALLE-3/
    │   ├── 0_real/
    │   └── 1_fake/
    └── ...
```

Supported extensions are `.jpg`, `.jpeg`, `.png`, and `.webp`.

## Installation

Activate the environment containing the correct CUDA-compatible PyTorch build:

```bash
conda activate conda_env
cd path/to/PIRE-DIRT
pip install -r requirements.txt
```

Editable installation is optional because both root entry points and DDP shell scripts automatically add `src/` to `PYTHONPATH`:

```bash
pip install -e .
```

## Configuration

Edit `configs/default.yaml` before training or testing:

```yaml
paths:
  train_root: /path/to/AIGIBench/train
  val_root: /path/to/AIGIBench/val
  test_root: /path/to/AIGIBench/test
  backbone_checkpoint: /path/to/model.safetensors
  trained_checkpoint: /path/to/trained_model.pth
  output_dir: outputs

model:
  dirt_hidden_dim: 256
  dirt_retain_ratio: 0.70
  use_pire: true
  pire_strength: 0.50
```

`dirt_retain_ratio` controls the proportion of important patch tokens retained by DIRT. PIRE is applied only during training, preserving the behavior of the original implementation.

## Three-GPU training

Recommended command:

```bash
cd /path/to/PIRE-DIRT
conda activate conda_env
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train_ddp.sh
```

The following direct command also works without installing the package:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
torchrun --standalone --nproc_per_node=3 \
  train.py --config configs/default.yaml
```

The best checkpoint is saved as:

```text
outputs/AIGIbench_best.pth
```

## Three-GPU testing

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/test_ddp.sh
```

Or:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
torchrun --standalone --nproc_per_node=3 \
  test.py --config configs/default.yaml
```

Testing is performed domain by domain. Results are saved to:

```text
outputs/test_results.csv
outputs/test_summary.json
```

## Passing command-line overrides

The shell scripts accept a configuration path followed by arbitrary `--set` overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train_ddp.sh \
  configs/default.yaml \
  --set train.epochs=20 \
  --set model.dirt_retain_ratio=0.2
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/test_ddp.sh \
  configs/default.yaml \
  --set paths.trained_checkpoint=/path/to/model.pth
```

## Single-GPU execution

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash scripts/train_ddp.sh
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash scripts/test_ddp.sh
```
