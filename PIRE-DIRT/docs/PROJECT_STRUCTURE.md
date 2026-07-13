# Project Structure

## Root entry points

- `train.py`: parses the YAML configuration, initializes DDP, and starts training.
- `test.py`: initializes DDP and evaluates every domain under the test root.

## `configs/`

- `default.yaml`: contains dataset paths, backbone/checkpoint paths, model settings, optimizer settings, dataloader settings, and distributed settings.

## `src/aigi_detector/data/`

- `transforms.py`: image duplication for small inputs, JPEG compression, and the standard normalization pipeline.
- `datasets.py`: dataset indexing, robust image loading, domain discovery, rank-based evaluation subsets, and collate helpers.

## `src/aigi_detector/models/`

- `lora.py`: QKV LoRA layer and automatic insertion into DINOv3 attention blocks.
- `fpsm.py`: style, frequency, and noise perturbation module.
- `mil.py`: Top-K multiple-instance learning head.
- `detector.py`: DINOv3 backbone and fusion of global and regional logits.

## `src/aigi_detector/engine/`

- `trainer.py`: distributed training loop, validation, gradient clipping, model selection, and checkpoint saving.
- `evaluator.py`: distributed inference, rank gathering, per-domain metrics, and result serialization.

## `src/aigi_detector/utils/`

- `config.py`: YAML loading and command-line overrides.
- `distributed.py`: process-group initialization, rank helpers, barriers, and reduction helpers.
- `seed.py`: deterministic random seeding and dataloader worker seeding.
- `checkpoint.py`: compatible checkpoint loading and standardized saving.
- `metrics.py`: accuracy and average precision.
- `logging.py`: rank-aware console output.

## Design changes from the original scripts

1. All hard-coded paths and hyperparameters are moved to YAML.
2. Training and testing import the same model implementation.
3. LoRA rank/alpha can no longer silently differ between train and test.
4. Validation runs on all DDP ranks rather than only rank 0.
5. Evaluation avoids `DistributedSampler` padding, so AP is not biased by duplicate samples.
6. Test results are exported to CSV and JSON.
7. The package can be installed with `pip install -e .`.
