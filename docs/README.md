# EM: 基于图神经网络的疾病相关分子关联预测框架

这是一个基于图神经网络（GNN）的框架，用于预测疾病相关的分子关联，包括lncRNA-疾病关联（LDA）、miRNA-疾病关联（MDA）和lncRNA-miRNA关联（LMI）。

## 目录结构

```
EM/
├── dataset1/                 # 数据集1
│   ├── disease_name.txt      # 疾病名称列表
│   ├── lncRNA_name.txt       # lncRNA名称列表
│   ├── lnc_dis.txt           # lncRNA-疾病关联
│   ├── miRNA-names.txt       # miRNA名称列表
│   └── mi_dis.txt            # miRNA-疾病关联
├── dataset2/                 # 数据集2
│   ├── lnc_dis.txt           # lncRNA-疾病关联
│   ├── lnc_mi.txt            # lncRNA-miRNA关联
│   └── mi_dis.txt            # miRNA-疾病关联
├── main.py                   # 主程序入口
├── train.py                  # 训练流程
├── data_preprocess.py        # 数据预处理
├── instantiation.py          # 模型实例化
├── layer.py                  # 模型层定义
├── parms_setting.py          # 参数设置
├── calculating_similarity.py # 相似性计算
├── utils.py                  # 工具函数
├── visualization.py          # 可视化工具
├── log_output_manager.py     # 日志和输出管理
├── FUSION_STRATEGY_USAGE.md  # 融合策略使用说明
├── TRI_ENTITY_FUSION_USAGE.md# 三实体融合使用说明
├── README_visualization.md   # 可视化说明
└── README.md                 # 本文件
```

## 环境要求

- Python 3.7+
- PyTorch 1.9+
- PyTorch Geometric
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Pandas

可以通过以下命令安装主要依赖：

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install numpy scipy scikit-learn matplotlib pandas
```

## 快速开始

### 1. 准备数据

项目包含两个数据集目录：
- `dataset1/`: 主要数据集
- `dataset2/`: 补充数据集

数据已经预先准备好了，可以直接使用。

### 2. 运行模型训练

使用默认参数运行模型：

```bash
python main.py
```

这将使用默认设置进行5折交叉验证训练，任务类型为LDA（lncRNA-疾病关联预测）。

### 3. 修改参数运行

可以根据需要修改参数，例如：

```bash
# 更改任务类型为MDA（miRNA-疾病关联）
python main.py --task_type MDA

# 更改任务类型为LMI（lncRNA-miRNA关联）
python main.py --task_type LMI

# 设置不同的特征类型
python main.py --feature_type one_hot

# 调整训练轮数
python main.py --epochs 100

# 设置学习率
python main.py --lr 0.001

# 使用GPU训练（如果有GPU）
python main.py --cuda_visible_devices 0
```

### 4. 常用参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task_type` | LDA | 任务类型：LDA（lncRNA-疾病）、MDA（miRNA-疾病）、LMI（lncRNA-miRNA） |
| `--feature_type` | normal | 特征类型：one_hot、uniform、normal、position |
| `--epochs` | 1 | 训练轮数 |
| `--lr` | 5e-4 | 学习率 |
| `--batch` | 25 | 批次大小 |
| `--dimensions` | 256 | 初始特征维度 |
| `--seed` | 0 | 随机种子 |

更多参数请查看 `parms_setting.py` 文件。

## 高级功能

### 融合策略

项目支持多种实体融合策略，包括：

1. **自注意力融合** (`self_attention`) - 默认策略，使用多头自注意力机制
2. **协作注意力融合** (`co_attention`) - 使用双向注意力机制让两个实体相互关注
3. **混合融合** (`hybrid`) - 结合上述两种策略，通过权重平衡

使用示例：
```bash
# 使用协作注意力融合
python main.py --fusion_strategy co_attention

# 使用混合融合，自注意力权重为0.7
python main.py --fusion_strategy hybrid --fusion_weight 0.7
```

详细说明请查看 [FUSION_STRATEGY_USAGE.md](FUSION_STRATEGY_USAGE.md)

### 三实体融合模式

除了传统的两实体关联预测，项目还支持三实体融合模式，可以同时考虑三个实体之间的复杂关系：

1. **三实体自注意力融合** (`tri_self_attention`) - 默认三实体策略
2. **三实体协作注意力融合** (`tri_co_attention`) - 使用循环注意力机制

使用示例：
```bash
# 启用三实体模式，使用默认策略
python main.py --tri_entity_mode True

# 启用三实体模式，使用协作注意力策略
python main.py --tri_entity_mode True --tri_fusion_strategy tri_co_attention
```

详细说明请查看 [TRI_ENTITY_FUSION_USAGE.md](TRI_ENTITY_FUSION_USAGE.md)

## 项目特点

### 1. 多任务学习框架
- 主任务：关联预测（监督学习）
- 对比学习任务：通过MoCo v2实现
- 对抗训练任务：提高模型鲁棒性

### 2. 图神经网络架构
- 编码器：GAT + Graph Transformer
- 融合模块：基于注意力机制的双实体/三实体融合
- 解码器：多层感知机

### 3. 数据增强策略
支持多种数据增强方法：
- `random_permute_features`：随机置换特征
- `attribute_mask`：属性遮蔽
- `noise_then_mask`：添加噪声后遮蔽

### 4. 交叉验证
采用5折交叉验证评估模型性能，确保结果可靠。

## 输出结果

运行完成后，结果将保存在 `OUTPUT/result/` 目录下，包括：

1. 模型性能指标：
   - AUROC（Area Under the ROC Curve）
   - AUPRC（Area Under the Precision-Recall Curve）
   - F1-score
   - 精确率和召回率

2. 可视化图表：
   - 训练/验证损失曲线
   - AUROC和AUPRC指标变化
   - ROC和PR曲线
   - 混淆矩阵
   - 校准曲线等

3. 详细日志：
   - 训练过程日志
   - 每一折的详细结果
   - 参数配置信息

## 可视化说明

项目提供了丰富的可视化功能，请参考 [README_visualization.md](README_visualization.md) 文件了解详细信息。

## 引用

如果您使用此代码，请引用相关论文（如果有的话）。

## 许可证

请查看LICENSE文件了解许可证信息。