# PIRE-DIRT 中文使用说明

## 1. 项目模块

该项目将训练与测试脚本拆分为标准 Python 工程，核心组成包括：

- DINOv3 ViT-L/16 主干网络；
- Attention QKV 上的 LoRA；
- CLS token 全局分类分支；
- **PIRE：Prior Information Region Enhancement**，从风格、频率和噪声统计三个方向增强区域特征；
- **DIRT：Dynamic Important Region Token filtering**，筛选重要 patch token 并形成区域预测；
- PyTorch DistributedDataParallel 多卡训练与测试。

## 2. 配置路径

打开 `configs/default.yaml`，至少修改：

```yaml
paths:
  train_root: 训练集路径
  val_root: 验证集路径
  test_root: 测试集路径
  backbone_checkpoint: DINOv3 权重路径
  trained_checkpoint: 测试 checkpoint 路径
  output_dir: 输出目录
```

数据格式为：

```text
split/domain_name/0_real/*
split/domain_name/1_fake/*
```

## 3. PIRE 与 DIRT 配置

```yaml
model:
  dirt_hidden_dim: 256
  dirt_retain_ratio: 0.10
  use_pire: true
  pire_strength: 0.20
```

- `dirt_retain_ratio`：DIRT 保留的重要区域 token 比例。
- `use_pire`：是否在训练阶段启用 PIRE。
- `pire_strength`：PIRE 对原始 patch token 的增强强度。

PIRE 仅在训练模式启用，验证和测试阶段不会添加随机扰动。

## 4. 环境安装

```bash
cd /opt/data/private/lh/PIRE-DIRT
conda activate qwen_vl
pip install -r requirements.txt
```

工程已经在 `train.py`、`test.py` 和运行脚本中自动加入 `src` 路径，因此不执行下面的安装也能运行：

```bash
pip install -e .
```

不过上传 GitHub 后，为了符合标准 Python 项目使用方式，仍建议执行一次可编辑安装。

## 5. 三卡训练

推荐运行：

```bash
cd /opt/data/private/lh/PIRE-DIRT
conda activate qwen_vl
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train_ddp.sh
```

原来的命令现在也可以直接使用：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
torchrun --standalone --nproc_per_node=3 \
  train.py --config configs/default.yaml
```

最佳模型默认保存为：

```text
outputs/AIGIbench_best.pth
```

## 6. 三卡测试

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/test_ddp.sh
```

测试会逐个子文件夹输出 ACC 和 AP，最后输出平均 ACC/AP，并保存：

```text
outputs/test_results.csv
outputs/test_summary.json
```

## 7. 临时覆盖参数

无需修改 YAML 即可覆盖参数：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train_ddp.sh \
  configs/default.yaml \
  --set train.epochs=20 \
  --set model.dirt_retain_ratio=0.2
```

## 8. 单卡运行

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash scripts/train_ddp.sh
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash scripts/test_ddp.sh
```

## 9. 旧模型兼容

测试代码会自动兼容旧版本训练得到的 checkpoint，包括：

- DDP 保存产生的 `module.` 前缀；
- 原始模型参数字段；
- `model` 或 `state_dict` 包装格式。

因此，原来的 `85.56.pth` 在 LoRA rank、网络维度和其他结构参数一致时仍然可以直接加载。

## 10. 实验协议提醒

默认配置中的 `val_root` 保留了原训练脚本指向 test 的设置，用于复现原逻辑。正式论文实验应使用独立验证集选择最佳 epoch，测试集只用于最终结果。
