# 中文使用说明

## 1. 项目用途

该项目将原始的训练与测试单文件脚本拆分为标准 Python 工程，核心方法保持为：

- DINOv3 ViT-L/16 主干网络；
- Attention QKV 上的 LoRA；
- CLS token 全局分类分支；
- Patch token Top-K MIL 区域分支；
- 仅训练阶段启用的 FPSM；
- PyTorch DistributedDataParallel 多卡训练与测试。

## 2. 首先修改配置

打开 `configs/default.yaml`，至少修改以下路径：

```yaml
paths:
  train_root: 训练集路径
  val_root: 验证集路径
  test_root: 测试集路径
  backbone_checkpoint: DINOv3 safetensors 权重路径
  trained_checkpoint: 测试时加载的模型路径
  output_dir: 输出目录
```

数据目录必须满足：

```text
split/domain_name/0_real/*
split/domain_name/1_fake/*
```

## 3. 安装

```bash
pip install -r requirements.txt
pip install -e .
```

应当先根据服务器 CUDA 版本安装匹配的 PyTorch 与 torchvision。

## 4. 三卡训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
torchrun --standalone --nproc_per_node=3 \
  train.py --config configs/default.yaml
```

也可以运行：

```bash
bash scripts/train_ddp.sh
```

最优模型默认保存为：

```text
outputs/AIGIbench_best.pth
```

## 5. 三卡测试

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
torchrun --standalone --nproc_per_node=3 \
  test.py --config configs/default.yaml
```

测试会逐个子文件夹输出 ACC 和 AP，最后输出所有子文件夹的平均 ACC/AP，并保存 CSV 与 JSON。

## 6. 与原代码相比的重要调整

- 训练与测试现在共用同一个模型文件，避免 LoRA rank、alpha 或网络结构不一致。
- 原训练代码只让 rank 0 使用 DDP 模型做验证，其他 rank 不参与，存在同步阻塞风险；新版本让所有 rank 共同验证。
- 测试仍采用手工切分索引，不使用会补齐样本的 `DistributedSampler`，避免重复样本影响 AP。
- 服务器路径全部放入 YAML，不再散落在代码中。
- checkpoint 兼容原始 `state_dict`、`model`、`state_dict` 字段以及 `module.` 前缀。

## 7. 实验协议提醒

默认配置中的 `val_root` 保留了原训练脚本指向 test 的行为，仅用于复现原逻辑。正式论文实验建议设置独立验证集，避免使用测试集选择最佳 epoch。
