# MoCo多视图配置使用指南

## 概述

本项目支持两种MoCo多视图实现，可以通过命令行参数灵活选择和配置：

1. **Basic MoCo** (`MoCoV2MultiView`) - 基础实现，简单高效
2. **Enhanced MoCo** (`EnhancedMoCoV2MultiView`) - 增强实现，带门控队列

## 快速开始

### 基础使用

```bash
# 使用基础MoCo（默认）
python main.py --moco_type basic

# 使用增强MoCo
python main.py --moco_type enhanced
```

### 高级配置

```bash
# 使用基础MoCo，自定义队列大小
python main.py --moco_type basic --moco_K 8192

# 使用增强MoCo，自定义门控参数
python main.py --moco_type enhanced --moco_gate_hidden 128 --moco_prune_threshold 1e-4

# 使用配置字符串（最灵活的方式）
python main.py --moco_config "enhanced[K=8192,gate_hidden=128,prune_threshold=0.0001]"
```

## 详细参数说明

### 基础MoCo参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--moco_type` | str | "basic" | MoCo类型：basic\|enhanced |
| `--moco_K` | int | 4096 | 队列大小 |
| `--moco_m` | float | 0.999 | 动量更新系数 |
| `--moco_T` | float | 0.2 | 温度系数 |
| `--moco_debug` | bool | False | 是否启用调试模式 |

### 增强MoCo特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--moco_gate_hidden` | int | 64 | 门控网络隐藏层维度 |
| `--moco_prune_threshold` | float | 1e-3 | 权重剪枝阈值 |
| `--moco_decay_factor` | float | 0.995 | 年龄衰减因子 |
| `--moco_use_learnable_queue_w` | bool | False | 使用可学习队列权重 |

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--moco_config` | str | None | 高级配置字符串 |
| `--queue_warmup_steps` | int | 0 | 队列预热步数 |

## 配置字符串格式

配置字符串格式为：`type[param1=value1,param2=value2,...]`

### 示例

```bash
# 基础MoCo，自定义队列和动量
python main.py --moco_config "basic[K=8192,m=0.995,T=0.1]"

# 增强MoCo，自定义所有门控参数
python main.py --moco_config "enhanced[K=4096,gate_hidden=128,prune_threshold=0.0001,decay_factor=0.99]"

# 启用可学习队列权重
python main.py --moco_config "enhanced[use_learnable_queue_w=true]"
```

## MoCo类型对比

| 特性 | Basic | Enhanced |
|------|--------|----------|
| **队列管理** | 基础环形队列 | 门控权重队列 |
| **温度控制** | 固定温度 | 自适应温度 |
| **负样本筛选** | 全量 | 全量+门控 |
| **计算复杂度** | 低(B×K) | 中等(B×K+门控计算) |
| **适用场景** | 一般对比学习 | 需要精细控制的场景 |
| **队列权重** | 无 | 可学习权重+年龄衰减 |
| **更新策略** | 动量更新 | 动量更新+GRU式门控 |

## 使用场景推荐

### 推荐使用Basic MoCo的场景：
- 标准对比学习任务
- 计算资源有限
- 初步实验和快速原型
- 大规模训练（需要高效率）

### 推荐使用Enhanced MoCo的场景：
- 需要精细控制负样本权重
- 数据分布复杂或噪声较多
- 需要自适应温度调整
- 研究对比学习的边界情况

## 完整示例

### 1. 基础实验

```bash
python main.py \
    --moco_type basic \
    --moco_K 4096 \
    --moco_T 0.2 \
    --fusion_strategy self_attention
```

### 2. 增强实验

```bash
python main.py \
    --moco_type enhanced \
    --moco_gate_hidden 128 \
    --moco_prune_threshold 1e-4 \
    --moco_decay_factor 0.99 \
    --attention_config gated_transformer[num_heads=8]
```

### 3. 自定义配置

```bash
python main.py \
    --moco_config "enhanced[K=8192,gate_hidden=256,prune_threshold=5e-4]" \
    --attention_config "gated_transformer[num_heads=8,headwise_gate=true]" \
    --moco_debug true
```

## 调试和监控

启用调试模式查看详细运行信息：

```bash
python main.py --moco_debug true
```

调试输出包括：
- 队列填充比例
- 余弦相似度统计
- 门控权重分布
- 温度调整情况
- 每个视图的详细信息

## 性能调优建议

1. **队列大小调整**：
   - 小数据集：K=1024-2048
   - 大数据集：K=4096-8192
   - 内存受限：减小K

2. **门控参数调整**：
   - 简单任务：gate_hidden=32-64
   - 复杂任务：gate_hidden=128-256
   - 噪声数据：减小prune_threshold

3. **温度参数**：
   - 标准情况：T=0.1-0.2
   - 困难任务：T=0.05-0.1
   - 简单任务：T=0.2-0.5

## 注意事项

1. **内存使用**：Enhanced MoCo由于门控网络会使用更多内存
2. **计算时间**：增强版本的计算时间会比基础版本增加15-30%
3. **收敛速度**：增强版本通常收敛更快，但每轮迭代时间更长
4. **参数敏感性**：增强版本的参数更敏感，建议谨慎调整

## 常见问题

### Q: 如何选择MoCo类型？
A: 建议先用basic版本进行快速实验，如果性能不理想再尝试enhanced版本。

### Q: 配置字符串中的参数优先级如何？
A: 配置字符串中的参数优先级高于单独的命令行参数。

### Q: 如何在训练过程中切换MoCo类型？
A: 不支持训练过程中动态切换，需要在启动前确定配置。

### Q: 两种MoCo类型可以混合使用吗？
A: 不建议混合使用，选择一种类型并保持训练一致性。