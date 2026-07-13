# DINOv3-LoRA-MIL-FPSM

A modular PyTorch implementation for AI-generated image detection using:

- a locally stored **DINOv3 ViT-L/16** backbone;
- **LoRA** adapters inserted into attention QKV projections;
- a global classification head based on the class token;
- a **Top-K MIL** head over patch tokens;
- an optional training-time **FPSM** feature perturbation module;
- distributed training and evaluation with PyTorch DDP.

This repository is reorganized from the original single-file training and testing scripts into reusable data, model, engine, configuration, and utility modules.

## Project structure

```text
dinov3_lora_mil_fpsm/
├── configs/
│   └── default.yaml
├── docs/
│   ├── PROJECT_STRUCTURE.md
│   └── README_zh-CN.md
├── scripts/
│   ├── test_ddp.sh
│   └── train_ddp.sh
├── src/aigi_detector/
│   ├── data/
│   ├── engine/
│   ├── models/
│   └── utils/
├── tests/
├── train.py
├── test.py
├── requirements.txt
└── pyproject.toml
```

## Dataset layout

The configured dataset root must contain one directory per domain or generator. Each domain contains real and fake images:

```text
AIGIBench/
├── train/
│   ├── progan/
│   │   ├── 0_real/
│   │   └── 1_fake/
│   └── sdv1.4/
│       ├── 0_real/
│       └── 1_fake/
├── val/                     # recommended for model selection
│   └── ...
└── test/
    ├── DALLE-3/
    │   ├── 0_real/
    │   └── 1_fake/
    └── ...
```

Supported image extensions are `.jpg`, `.jpeg`, `.png`, and `.webp`.

## Installation

```bash
git clone <your-repository-url>
cd dinov3_lora_mil_fpsm

conda create -n aigi-detector python=3.10 -y
conda activate aigi-detector

# Install a PyTorch build compatible with your CUDA environment first.
pip install -r requirements.txt
pip install -e .
```

## Configuration

Edit `configs/default.yaml` before running. At minimum, set:

```yaml
paths:
  train_root: /path/to/AIGIBench/train
  val_root: /path/to/AIGIBench/val
  test_root: /path/to/AIGIBench/test
  backbone_checkpoint: /path/to/model.safetensors
  trained_checkpoint: outputs/AIGIbench_best.pth
```

The model settings used by training and testing are read from the same file, avoiding LoRA-rank or head-configuration mismatches.

## Distributed training

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
torchrun --standalone --nproc_per_node=3 \
  train.py --config configs/default.yaml
```

Or:

```bash
bash scripts/train_ddp.sh
```

The best checkpoint is selected using validation accuracy and saved under `paths.output_dir`. Validation is distributed across all ranks, so every DDP process participates in evaluation.

## Distributed testing

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
torchrun --standalone --nproc_per_node=3 \
  test.py --config configs/default.yaml
```

Or:

```bash
bash scripts/test_ddp.sh
```

Testing is performed domain by domain. Samples are partitioned manually across ranks without sampler padding, preventing duplicated samples from affecting AP. The script reports per-domain ACC/AP and macro averages across domains, then writes:

```text
outputs/test_results.csv
outputs/test_summary.json
```

## Single-GPU execution

The same entry points also work with one process:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  train.py --config configs/default.yaml

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  test.py --config configs/default.yaml
```

## Checkpoint compatibility

The loader accepts:

- a raw model `state_dict`;
- a dictionary containing `model`;
- a dictionary containing `state_dict`;
- keys prefixed with `module.` from DDP training.

Existing checkpoints from the original scripts remain loadable when model hyperparameters match.

## Notes

1. The original logic applies FPSM only while `model.training == True`; therefore FPSM acts as a training-time feature perturbation module and is disabled during validation/testing.
2. `val_root` should normally point to a validation split. Pointing it to the test split reproduces the original training script but is not recommended for a publishable experimental protocol.
3. Large model weights, datasets, and generated checkpoints are ignored by Git and should not be uploaded directly to the repository.

## Publish to GitHub

```bash
git init
git add .
git commit -m "Initial release"
git branch -M main
git remote add origin <your-github-repository-url>
git push -u origin main
```

Before pushing, verify that datasets, DINOv3 weights, and trained `.pth` files are not staged. They are excluded by `.gitignore`.

## License

No license is included automatically. Add an appropriate license before making the repository public.
