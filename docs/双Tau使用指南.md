# MoCoV2MultiViewDoubleτ 使用指南

`MoCoV2MultiViewDoubleτ` 是 MoCo v2 多视图模型的双温度参数版本，使用 τ₁ 控制类内吸引强度，τ₂ 控制类间排斥强度。

## 核心特性

- **双温度参数**：τ₁ 控制正样本相似度缩放，τ₂ 控制负样本相似度缩放
- **独立控制**：可以精细调整类内吸引和类间排斥的强度
- **向后兼容**：当 τ₂ 未指定时，自动使用 τ₁ 作为统一温度参数
- **队列管理**：保持 MoCo 的环形队列机制，支持大规模负样本存储

## 命令行使用方法

### 基础用法
```bash
python main.py --moco_type double_tau
```

### 自定义温度参数
```bash
python main.py \
    --moco_type double_tau \
    --moco_tau1 0.15 \
    --moco_tau2 0.25 \
    --epochs 100 \
    --seed 42 \
    --run_name double_tau_experiment
```

### 使用配置字符串
```bash
python main.py \
    --moco_config "double_tau[K=8192,m=0.999,tau1=0.2,tau2=0.3]" \
    --epochs 200 \
    --seed 0 \
    --run_name custom_double_tau
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--moco_type` | str | basic | MoCo类型，选择 `double_tau` 启用双温度模式 |
| `--moco_tau1` | float | 0.2 | 正样本温度系数(τ₁)，控制类内吸引强度 |
| `--moco_tau2` | float | 0.3 | 负样本温度系数(τ₂)，控制类间排斥强度 |
| `--moco_K` | int | 4096 | 队列大小，存储负样本的数量 |
| `--moco_m` | float | 0.999 | 动量更新系数 |
| `--queue_warmup_steps` | int | 0 | 队列预热步数 |

## 与基础 MoCo 的对比

| 特性 | Basic MoCo | DoubleTau MoCo |
|------|------------|---------------|
| 温度控制 | 固定温度 T | 双温度 τ₁, τ₂ |
| 正样本缩放 | T | τ₁ |
| 负样本缩放 | T | τ₂ |
| 参数调优 | 单一温度 | 独立精细调优 |
| 适用场景 | 一般对比学习 | 需要精细控制的任务 |

## 推荐参数设置

### 保守设置（适用于稳定训练）
- τ₁ = 0.2（适度吸引）
- τ₂ = 0.3（适度排斥）

### 激进设置（适用于困难样本）
- τ₁ = 0.1（强吸引）
- τ₂ = 0.5（强排斥）

### 平衡设置
- τ₁ = 0.15
- τ₂ = 0.25

## 技术细节

### 前向传播
```python
# 正样本相似度
l_pos = torch.sum(q * k, dim=1, keepdim=True) / self.tau1

# 负样本相似度  
l_neg = torch.matmul(q, queue.clone().detach()) / self.tau2

# 合并logits
logits = torch.cat([l_pos, l_neg], dim=1)
```

### 参数验证
- τ₁ 必须大于 0
- τ₂ 必须大于 0（如果提供）
- 推荐范围：0.05 - 1.0

## 使用示例

### 场景1：一般对比学习
```bash
python main.py --moco_type double_tau --moco_tau1 0.2 --moco_tau2 0.3
```

### 场景2：细粒度分类
```bash
python main.py --moco_type double_tau --moco_tau1 0.15 --moco_tau2 0.35
```

### 场景3：大规模预训练
```bash
python main.py --moco_type double_tau --moco_K 8192 --moco_tau1 0.25 --moco_tau2 0.4
```

## 注意事项

1. **温度参数范围**：τ₁ 和 τ₂ 建议在 0.05-1.0 范围内
2. **参数关系**：通常 τ₂ ≥ τ₁，确保负样本排斥不过度
3. **训练稳定性**：极端值可能导致训练不稳定
4. **任务适配**：不同任务需要调整温度参数以获得最佳效果

## 故障排除

### 问题1：损失值异常
- 检查 τ₁, τ₂ 是否在合理范围
- 确认输入特征已正确归一化

### 问题2：训练不收敛
- 尝试调整温度参数比例
- 检查学习率是否合适

### 问题3：内存不足
- 减少 K（队列大小）
- 调整批处理大小

这个实现为对比学习提供了更精细的控制，通过独立调节正负样本的温度参数，可以适应不同的学习任务和数据特性。