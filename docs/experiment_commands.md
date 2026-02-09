# EM项目实验命令行配置方案

根据控制变量法原则，设计以下实验以验证不同模块的效果。

## 1. Baseline实验（对照组）

```bash
# 默认Baseline配置
python main.py --seed=0 --epochs 10 --run_name baseline --fusion_strategy self_attention --moco_type basic --use_co_attention false --tri_entity_mode false
```

## 2. 协作注意力类型实验

```bash
# Baseline + Pairwise协作注意力
python main.py --seed=0 --epochs 10 --run_name co_attention_pairwise --fusion_strategy co_attention --co_attention_type pairwise --moco_type basic --tri_entity_mode false

# Baseline + Transformer协作注意力
python main.py --seed=0 --epochs 10 --run_name co_attention_transformer --fusion_strategy co_attention --co_attention_type transformer --moco_type basic --tri_entity_mode false

# Baseline + Multihead协作注意力
python main.py --seed=0 --epochs 10 --run_name co_attention_multihead --attention_config "co_attention[type=multihead,num_heads=4]" --moco_type basic --tri_entity_mode false

# Baseline + Gated Multihead协作注意力
python main.py --seed=0 --epochs 10 --run_name co_attention_gated_multihead --attention_config "co_attention[type=gated_multihead,num_heads=4,headwise_gate=true]" --moco_type basic --tri_entity_mode false
```

## 3. 融合策略类型实验

```bash
# 自注意力融合（Baseline）
python main.py --seed=0 --epochs 10 --run_name fusion_self_attention --fusion_strategy self_attention --moco_type basic --tri_entity_mode false

# 协作注意力融合
python main.py --seed=0 --epochs 10 --run_name fusion_co_attention --fusion_strategy co_attention --co_attention_type transformer --moco_type basic --tri_entity_mode false

# 混合融合（70%自注意力）
python main.py --seed=0 --epochs 10 --run_name fusion_hybrid_0.7 --fusion_strategy hybrid --fusion_weight 0.7 --co_attention_type transformer --moco_type basic --tri_entity_mode false

# 混合融合（30%自注意力）
python main.py --seed=0 --epochs 10 --run_name fusion_hybrid_0.3 --fusion_strategy hybrid --fusion_weight 0.3 --co_attention_type transformer --moco_type basic --tri_entity_mode false

# 门控Transformer融合
python main.py --seed=0 --epochs 10 --run_name fusion_gated_transformer --attention_config "gated_transformer[num_heads=8,headwise_gate=true]" --moco_type basic --tri_entity_mode false
```

## 4. 门控机制实验

```bash
# 无门控（Baseline）
python main.py --seed=0 --epochs 10 --run_name gating_none --fusion_strategy self_attention --moco_type basic --tri_entity_mode false

# 头级门控
python main.py --seed=0 --epochs 10 --run_name gating_headwise --attention_config "gated_transformer[num_heads=8,headwise_gate=true]" --moco_type basic --tri_entity_mode false

# 元素级门控
python main.py --seed=0 --epochs 10 --run_name gating_elementwise --attention_config "gated_transformer[num_heads=8,elementwise_gate=true]" --moco_type basic --tri_entity_mode false

# 头级+元素级门控
python main.py --seed=0 --epochs 10 --run_name gating_combined --attention_config "gated_transformer[num_heads=8,headwise_gate=true,elementwise_gate=true]" --moco_type basic --tri_entity_mode false
```

## 5. MoCo多视图对比学习类型实验

```bash
# 基础MoCo（Baseline）
python main.py --seed=0 --epochs 10 --run_name moco_basic --moco_type basic --moco_K 4096 --fusion_strategy self_attention --tri_entity_mode false

# 增强MoCo
python main.py --seed=0 --epochs 10 --run_name moco_enhanced_default --moco_type enhanced --moco_gate_hidden 64 --moco_prune_threshold 1e-3 --fusion_strategy self_attention --tri_entity_mode false

# 增强MoCo（大隐藏层）
python main.py --seed=0 --epochs 10 --run_name moco_enhanced_large_hidden --moco_type enhanced --moco_gate_hidden 128 --moco_prune_threshold 1e-3 --fusion_strategy self_attention --tri_entity_mode false

# 增强MoCo（低剪枝阈值）
python main.py --seed=0 --epochs 10 --run_name moco_enhanced_low_prune --moco_type enhanced --moco_gate_hidden 64 --moco_prune_threshold 1e-4 --fusion_strategy self_attention --tri_entity_mode false

# 自定义配置增强MoCo
python main.py --seed=0 --epochs 10 --run_name moco_enhanced_custom --moco_config "enhanced[K=8192,gate_hidden=128,prune_threshold=5e-4]" --fusion_strategy self_attention --tri_entity_mode false
```

## 6. 三实体融合实验

```bash
# 两实体模式（Baseline）
python main.py --seed=0 --epochs 10 --run_name tri_entity_false --tri_entity_mode false --fusion_strategy self_attention --moco_type basic

# 三实体自注意力融合
python main.py --seed=0 --epochs 10 --run_name tri_entity_self_attention --tri_entity_mode true --tri_fusion_strategy tri_self_attention --moco_type basic

# 三实体协作注意力融合
python main.py --seed=0 --epochs 10 --run_name tri_entity_co_attention --tri_entity_mode true --tri_fusion_strategy tri_co_attention --co_hidden_dim 128 --moco_type basic
```

## 7. 综合最优配置实验

```bash
# 综合最优配置
python main.py --seed=0 --epochs 10 --run_name optimal_combined --tri_entity_mode true --tri_fusion_strategy tri_co_attention --co_hidden_dim 256 --moco_config "enhanced[K=8192,gate_hidden=128,prune_threshold=5e-4]" --attention_config "gated_transformer[num_heads=8,headwise_gate=true]"
```

## 8. 不同任务类型实验

```bash
# LDA任务Baseline
python main.py --seed=0 --epochs 10 --run_name task_lda_baseline --task_type LDA --fusion_strategy self_attention --moco_type basic --tri_entity_mode false

# MDA任务Baseline
python main.py --seed=0 --epochs 10 --run_name task_mda_baseline --task_type MDA --fusion_strategy self_attention --moco_type basic --tri_entity_mode false

# LMI任务Baseline
python main.py --seed=0 --epochs 10 --run_name task_lmi_baseline --task_type LMI --fusion_strategy self_attention --moco_type basic --tri_entity_mode false
```

以上实验设计遵循控制变量法原则，每次只改变一个变量，保持其他条件一致，从而能够准确评估各个模块的效果。