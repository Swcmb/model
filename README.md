# EM: 基于图神经网络的疾病相关分子关联预测框架

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 项目简介

EM（Embedding Model）是一个基于图神经网络（GNN）的深度学习框架，专门用于预测生物医学领域中的分子关联，包括：

- **LDA**（lncRNA-疾病关联预测）
- **MDA**（miRNA-疾病关联预测）
- **LMI**（lncRNA-miRNA关联预测）

本项目融合了多项先进技术，包括图注意力网络、图变换器、多视图对比学习、注意力融合机制和对抗训练，在多个基准数据集上取得了优异的性能。

## ✨ 核心特性

### 🧠 先进的模型架构

- **GAT + Graph Transformer 编码器**：结合图注意力网络和图变换器的优势，有效捕获节点特征和图结构信息
- **多注意力融合机制**：支持自注意力、协作注意力、混合注意力等多种融合策略
- **三实体融合模式**：支持同时建模三个实体之间的复杂交互关系

### 🔄 多视图对比学习

- **MoCo v2 / BYOL**：支持两种主流的自监督学习框架
- **多视图机制**：通过数据增强生成多个视图，增强模型鲁棒性
- **增强MoCo**：支持门控权重、自适应温度等高级特性

### ⚔️ 对抗训练

- **FGSM / PGD 攻击**：支持快速梯度符号法和投影梯度下降攻击
- **多预算扰动**：支持共享预算和独立预算两种模式
- **节点级对抗损失**：提升模型对输入扰动的鲁棒性

### 📊 数据增强策略

- **随机特征置换**：随机打乱节点特征顺序
- **属性掩蔽**：随机掩蔽部分特征维度
- **噪声掩蔽**：添加高斯噪声后进行掩蔽
- **在线/离线增强**：支持静态增强和在线动态增强

### 🎨 完整的可视化系统

- **训练曲线监控**：实时监控训练/验证损失和性能指标
- **ROC/PR曲线**：评估分类性能
- **混淆矩阵**：可视化预测结果
- **概率校准**：评估预测概率的可靠性
- **温度缩放**：优化概率校准效果

## 📁 项目结构

```
EM/
├── dataset1/                          # 主要数据集
│   ├── disease_name.txt               # 疾病名称列表
│   ├── lncRNA_name.txt                # lncRNA名称列表
│   ├── miRNA-names.txt                # miRNA名称列表
│   ├── LDA.edgelist                   # lncRNA-疾病关联边列表
│   ├── MDA.edgelist                   # miRNA-疾病关联边列表
│   ├── LMI.edgelist                   # lncRNA-miRNA关联边列表
│   ├── non_LDA.edgelist               # LDA负样本
│   ├── non_MDA.edgelist               # MDA负样本
│   ├── non_LMI.edgelist               # LMI负样本
│   ├── dis_fuse_sim_0.8.txt           # 疾病融合相似度
│   ├── lnc_fuse_sim_0.8.txt           # lncRNA融合相似度
│   ├── mi_fuse_sim_0.8.txt            # miRNA融合相似度
│   └── ...                            # 其他相似度文件
├── dataset2/                          # 补充数据集（结构同dataset1）
├── main.py                            # 主程序入口
├── train.py                           # 训练流程
├── layer.py                           # 模型层定义（核心架构）
├── instantiation.py                   # 模型实例化
├── data_preprocess.py                 # 数据预处理
├── calculating_similarity.py          # 相似度计算
├── parms_setting.py                   # 参数设置
├── utils.py                           # 工具函数
├── visualization.py                   # 可视化工具
├── log_output_manager.py              # 日志和输出管理
├── run_experiments.py                 # 实验运行脚本
├── test_byol_integration.py           # BYOL集成测试
└── docs/                              # 文档目录
    ├── README.md                      # 本文件
    ├── 项目实现细节.md                # 技术实现细节
    ├── 综合系统指南.md                # 系统使用指南
    ├── 注意力机制概述.md              # 注意力机制说明
    ├── 融合策略使用指南.md            # 融合策略详细说明
    ├── 三实体融合使用指南.md          # 三实体融合说明
    ├── MoCo类型使用指南.md            # MoCo配置说明
    ├── 可视化说明.md                  # 可视化功能说明
    └── experiment_commands.md         # 实验命令示例
```

## 🚀 快速开始

### 环境要求

- Python 3.7+
- PyTorch 1.9+
- PyTorch Geometric
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Pandas

### 安装依赖

```bash
# 安装PyTorch（根据您的CUDA版本选择）
pip install torch torchvision torchaudio

# 安装PyTorch Geometric
pip install torch-geometric

# 安装其他依赖
pip install numpy scipy scikit-learn matplotlib pandas
```

### 基础使用

#### 1. 运行默认配置

```bash
python main.py
```

这将使用默认设置进行5折交叉验证训练，任务类型为LDA。

#### 2. 更改任务类型

```bash
# MDA任务（miRNA-疾病关联）
python main.py --task_type MDA

# LMI任务（lncRNA-miRNA关联）
python main.py --task_type LMI
```

#### 3. 调整训练参数

```bash
# 增加训练轮数
python main.py --epochs 100

# 调整学习率
python main.py --lr 0.001

# 修改批次大小
python main.py --batch 64

# 调整模型维度
python main.py --dimensions 512 --hidden1 256 --hidden2 128
```

#### 4. 使用GPU

```bash
# 指定GPU设备
python main.py --cuda_visible_devices 0
```

## 🎯 高级功能

### 注意力融合策略

项目支持多种融合策略，可通过 `--fusion_strategy` 参数配置：

```bash
# 自注意力融合（默认）
python main.py --fusion_strategy self_attention

# 协作注意力融合
python main.py --fusion_strategy co_attention

# 混合融合
python main.py --fusion_strategy hybrid --fusion_weight 0.7

# Transformer多头注意力
python main.py --fusion_strategy transformer_multihead
```

### 三实体融合模式

```bash
# 启用三实体模式
python main.py --tri_entity_mode True

# 三实体自注意力
python main.py --tri_entity_mode True --tri_fusion_strategy tri_self_attention

# 三实体协作注意力
python main.py --tri_entity_mode True --tri_fusion_strategy tri_co_attention
```

### 对比学习配置

#### MoCo类型

```bash
# 基础MoCo
python main.py --model_type moco --moco_type basic

# DoubleTau MoCo
python main.py --model_type moco --moco_type double_tau

# 调整MoCo参数
python main.py --moco_K 4096 --moco_m 0.999 --moco_T 0.2
```

#### BYOL

```bash
# 使用BYOL
python main.py --model_type byol

# 调整BYOL参数
python main.py --byol_predictor_dim 256 --byol_ema_momentum 0.996
```

### 对抗训练

```bash
# 启用对抗训练
python main.py --adv_mode mgraph

# PGD攻击
python main.py --adv_mode mgraph --adv_eps 0.01 --adv_alpha 0.005 --adv_steps 3

# FGSM攻击
python main.py --adv_mode mgraph --adv_eps 0.01 --adv_steps 1

# 调整对抗预算
python main.py --adv_mode mgraph --adv_budget independent
```

### 数据增强

```bash
# 离线增强（默认）
python main.py --augment_mode static

# 在线增强
python main.py --augment_mode online

# 选择增强方式
python main.py --augment random_permute_features,attribute_mask,noise_then_mask

# 调整增强参数
python main.py --noise_std 0.01 --mask_rate 0.1
```

## 📊 参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task_type` | LDA | 任务类型：LDA/MDA/LMI |
| `--epochs` | 1 | 训练轮数 |
| `--lr` | 5e-4 | 学习率 |
| `--batch` | 25 | 批次大小 |
| `--dimensions` | 256 | 初始特征维度 |
| `--hidden1` | 128 | 编码器第一隐藏层维度 |
| `--hidden2` | 64 | 编码器第二隐藏层维度 |
| `--decoder1` | 512 | 解码器隐藏层维度 |
| `--dropout` | 0.1 | Dropout比例 |
| `--weight_decay` | 5e-4 | 权重衰减系数 |

### 损失权重

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--loss_ratio1` | 1.0 | 监督任务权重 |
| `--loss_ratio2` | 0.5 | 对比学习任务权重 |
| `--loss_ratio3` | 0.5 | 节点对抗任务权重 |

### 模型架构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gat_heads` | 4 | GAT注意力头数 |
| `--gt_heads` | 4 | Graph Transformer注意力头数 |
| `--fusion_heads` | 4 | 融合模块注意力头数 |
| `--moco_K` | 4096 | MoCo队列大小 |
| `--moco_m` | 0.999 | MoCo动量更新系数 |
| `--moco_T` | 0.2 | MoCo温度系数 |

### 数据增强参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--feature_type` | normal | 特征类型：one_hot/uniform/normal/position |
| `--augment_mode` | static | 增强模式：static/online |
| `--noise_std` | 0.01 | 高斯噪声标准差 |
| `--mask_rate` | 0.1 | 特征掩蔽比例 |

### 对抗训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--adv_mode` | none | 对抗模式：none/mgraph |
| `--adv_norm` | linf | 对抗范数：linf/l2 |
| `--adv_eps` | 0.01 | 对抗扰动预算 |
| `--adv_alpha` | 0.005 | PGD单步步长 |
| `--adv_steps` | 0 | PGD步数 |

完整参数列表请运行 `python main.py --help` 查看。

## 📈 输出结果

运行完成后，结果保存在 `OUTPUT/result/` 目录下，包括：

### 1. 性能指标

- **AUROC**（ROC曲线下面积）
- **AUPRC**（PR曲线下面积）
- **F1-Score**
- **精确率**（Precision）
- **召回率**（Recall）
- **混淆矩阵**

### 2. 可视化图表

- `epoch_curves_fold_*.png` - 训练/验证损失与AUROC曲线
- `loss_breakdown_fold_*.png` - 损失分解图
- `epoch_metrics_bar_fold_*.png` - 每epoch性能指标
- `roc_fold_*.png` - ROC曲线
- `pr_fold_*.png` - PR曲线
- `calibration_fold_*.png` - 概率校准曲线
- `threshold_scan_fold_*.png` - 阈值扫描结果
- `confusion_matrix_sum.png` - 混淆矩阵热力图

### 3. 详细日志

- 训练过程日志
- 每一折的详细结果
- 参数配置信息
- 对抗配置信息

## 📚 详细文档

项目包含详细的技术文档，请查看 `docs/` 目录：

- **[项目实现细节.md](docs/项目实现细节.md)** - 详细的技术实现说明
- **[综合系统指南.md](docs/综合系统指南.md)** - 完整的系统使用指南
- **[注意力机制概述.md](docs/注意力机制概述.md)** - 注意力机制详解
- **[融合策略使用指南.md](docs/融合策略使用指南.md)** - 融合策略详细说明
- **[三实体融合使用指南.md](docs/三实体融合使用指南.md)** - 三实体融合说明
- **[MoCo类型使用指南.md](docs/MOCO类型使用指南.md)** - MoCo配置说明
- **[可视化说明.md](docs/可视化说明.md)** - 可视化功能说明
- **[experiment_commands.md](docs/experiment_commands.md)** - 实验命令示例

## 🎓 使用示例

### 示例1：基线实验

```bash
# 两实体基础配置
python main.py \
  --task_type LDA \
  --fusion_strategy self_attention \
  --model_type moco \
  --epochs 100 \
  --run_name "baseline_2entity"
```

### 示例2：注意力机制对比

```bash
# Transformer多头注意力
python main.py \
  --fusion_strategy transformer_multihead \
  --fusion_heads 8 \
  --epochs 100 \
  --run_name "transformer_8heads"

# 协作注意力
python main.py \
  --fusion_strategy co_attention \
  --co_attention_type transformer \
  --epochs 100 \
  --run_name "coattention_transformer"
```

### 示例3：三实体融合

```bash
# 三实体自注意力
python main.py \
  --tri_entity_mode True \
  --tri_fusion_strategy tri_self_attention \
  --epochs 100 \
  --run_name "tri_self_attention"

# 三实体协作注意力
python main.py \
  --tri_entity_mode True \
  --tri_fusion_strategy tri_co_attention \
  --epochs 100 \
  --run_name "tri_coattention"
```

### 示例4：对抗训练

```bash
# PGD对抗训练
python main.py \
  --adv_mode mgraph \
  --adv_eps 0.01 \
  --adv_alpha 0.005 \
  --adv_steps 3 \
  --adv_warmup_end 10 \
  --epochs 100 \
  --run_name "adversarial_training"
```

### 示例5：高级配置组合

```bash
# 最强配置组合
python main.py \
  --tri_entity_mode True \
  --tri_fusion_strategy tri_co_attention \
  --fusion_strategy gated_transformer \
  --fusion_heads 8 \
  --model_type moco \
  --moco_type double_tau \
  --adv_mode mgraph \
  --adv_eps 0.01 \
  --adv_steps 3 \
  --epochs 200 \
  --run_name "full_enhanced_system"
```

## 🔧 故障排除

### 常见问题

#### 1. 内存不足（OOM）

```bash
# 减少批大小
python main.py --batch 32

# 减少模型维度
python main.py --hidden1 128 --hidden2 64

# 减少MoCo队列大小
python main.py --moco_K 1024
```

#### 2. 训练速度慢

```bash
# 减少注意力头数
python main.py --fusion_heads 4

# 使用基础融合策略
python main.py --fusion_strategy self_attention

# 关闭对抗训练
python main.py --adv_mode none
```

#### 3. CUDA相关错误

```bash
# 检查CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"

# 使用CPU训练
python main.py --cuda_visible_devices ""
```

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢所有为本项目做出贡献的研究人员和开发者。

## 📮 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/yourusername/EM/issues)
- 发送邮件至：your.email@example.com

## 📝 更新日志

### v1.0.0 (2025-01-10)
- 初始版本发布
- 支持LDA、MDA、LMI三种任务
- 实现GAT+Graph Transformer编码器
- 支持多种注意力融合策略
- 集成MoCo v2和BYOL对比学习
- 添加对抗训练功能
- 完整的可视化系统

---

**最后更新**: 2025年1月10日
**版本**: v1.0