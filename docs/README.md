# PIRE-DIRT User Guide

## 1. Project Modules

This project organizes the training and testing scripts as a standard Python project. Its core components include:

- DINOv3 ViT-L/16 backbone;
- LoRA applied to the attention QKV projections;
- A global classification branch based on the CLS token;
- **PIRE: Prior Information Region Enhancement**, which enhances regional features from three perspectives: style, frequency, and noise statistics;
- **DIRT: Dynamic Important Region Token Filtering**, which selects important patch tokens and produces regional predictions;
- Multi-GPU training and testing with PyTorch DistributedDataParallel.

## 2. Path Configuration

Open `configs/default.yaml` and modify at least the following paths:

```yaml
paths:
  train_root: path/to/training/set
  val_root: path/to/validation/set
  test_root: path/to/test/set
  backbone_checkpoint: path/to/DINOv3/checkpoint
  trained_checkpoint: path/to/trained/checkpoint/for/testing
  output_dir: path/to/output/directory
```

The expected dataset structure is:

```text
split/domain_name/0_real/*
split/domain_name/1_fake/*
```

## 3. PIRE and DIRT Configuration

```yaml
model:
  dirt_hidden_dim: 256
  dirt_retain_ratio: 0.10
  use_pire: true
  pire_strength: 0.20
```

- `dirt_retain_ratio`: The proportion of important regional tokens retained by DIRT.
- `use_pire`: Whether to enable PIRE during training.
- `pire_strength`: The enhancement strength applied by PIRE to the original patch tokens.

PIRE is enabled only in training mode. No random perturbations are added during validation or testing.

## 4. Environment Setup

```bash
cd /opt/data/private/lh/PIRE-DIRT
conda activate qwen_vl
pip install -r requirements.txt
```

The `src` path is automatically added in `train.py`, `test.py`, and the execution scripts, so the project can run without performing the following installation:

```bash
pip install -e .
```

However, after uploading the project to GitHub, an editable installation is still recommended to follow standard Python project usage.

## 5. Training with Three GPUs

The recommended command is:

```bash
cd /opt/data/private/lh/PIRE-DIRT
conda activate qwen_vl
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train_ddp.sh
```

The original command can also be used directly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
torchrun --standalone --nproc_per_node=3 \
  train.py --config configs/default.yaml
```

By default, the best model is saved as:

```text
outputs/AIGIbench_best.pth
```

## 6. Testing with Three GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/test_ddp.sh
```

The test script reports ACC and AP for each subfolder, followed by the mean ACC and AP across all subsets. The results are saved to:

```text
outputs/test_results.csv
outputs/test_summary.json
```

## 7. Temporarily Overriding Configuration Parameters

Parameters can be overridden without modifying the YAML file:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train_ddp.sh \
  configs/default.yaml \
  --set train.epochs=20 \
  --set model.dirt_retain_ratio=0.2
```

## 8. Single-GPU Execution

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash scripts/train_ddp.sh
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash scripts/test_ddp.sh
```

## 9. Compatibility with Legacy Models

The testing code automatically supports checkpoints produced by earlier versions, including:

- The `module.` prefix introduced by DDP checkpoint saving;
- The original model parameter fields;
- Checkpoints wrapped in either `model` or `state_dict` format.

Therefore, the original `85.56.pth` checkpoint can still be loaded directly, provided that the LoRA rank, network dimensions, and other architecture-related parameters remain consistent.

## 10. Experimental Protocol Reminder

In the default configuration, `val_root` retains the original training-script setting that points to the test set in order to reproduce the original workflow. For formal paper experiments, an independent validation set should be used to select the best epoch, while the test set should be used only for final evaluation.
