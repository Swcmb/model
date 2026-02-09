# 控制变量法测试模型类型实验命令

## 1. Baseline实验（对照组）

```bash
# 默认Baseline配置
python main.py --seed=0 --epochs 10 --run_name baseline --fusion_strategy self_attention --moco_type basic --use_co_attention false
```

## 2. 协作注意力类型实验（控制其他变量不变，只改变协作注意力类型）

```bash
# Baseline + Pairwise协作注意力
python main.py --seed=0 --epochs 10 --run_name co_attention_pairwise --fusion_strategy co_attention --co_attention_type pairwise --moco_type basic

# Baseline + Transformer协作注意力
python main.py --seed=0 --epochs 10 --run_name co_attention_transformer --fusion_strategy co_attention --co_attention_type transformer --moco_type basic

# Baseline + Multihead协作注意力
python main.py --seed=0 --epochs 10 --run_name co_attention_multihead --attention_config "co_attention[type=multihead,num_heads=4]" --moco_type basic
```

## 3. 融合策略类型实验（控制其他变量不变，只改变融合策略）

```bash
# 自注意力融合（Baseline）
python main.py --seed=0 --epochs 10 --run_name fusion_self_attention --fusion_strategy self_attention --moco_type basic

# 协作注意力融合
python main.py --seed=0 --epochs 10 --run_name fusion_co_attention --fusion_strategy co_attention --co_attention_type transformer --moco_type basic

# 混合融合（70%自注意力）
python main.py --seed=0 --epochs 10 --run_name fusion_hybrid_0.7 --fusion_strategy hybrid --fusion_weight 0.7 --co_attention_type transformer --moco_type basic

# 混合融合（30%自注意力）
python main.py --seed=0 --epochs 10 --run_name fusion_hybrid_0.3 --fusion_strategy hybrid --fusion_weight 0.3 --co_attention_type transformer --moco_type basic
```

## 4. 自监督学习模型类型实验（控制其他变量不变，只改变模型类型）

```bash
# MoCo模型（默认）
python main.py --seed=0 --epochs 10 --run_name moco_baseline --model_type moco --fusion_strategy self_attention --moco_type basic

# BYOL模型
python main.py --seed=0 --epochs 10 --run_name byol_baseline --model_type byol --fusion_strategy self_attention

# BYOL模型（不同EMA动量）
python main.py --seed=0 --epochs 10 --run_name byol_ema_0998 --model_type byol --fusion_strategy self_attention --byol_ema_momentum 0.998

# BYOL模型（不同预测头维度）
python main.py --seed=0 --epochs 10 --run_name byol_predictor_512 --model_type byol --fusion_strategy self_attention --byol_predictor_dim 512
```

## 5. MoCo变体实验（控制其他变量不变，只改变MoCo配置）

```bash
# Basic MoCo（Baseline）
python main.py --seed=0 --epochs 10 --run_name moco_basic --moco_type basic --fusion_strategy self_attention

# Enhanced MoCo（默认配置）
python main.py --seed=0 --epochs 10 --run_name moco_enhanced_default --moco_type enhanced --moco_gate_hidden 64 --moco_prune_threshold 1e-3 --fusion_strategy self_attention

# Enhanced MoCo（大隐藏层）
python main.py --seed=0 --epochs 10 --run_name moco_enhanced_large_hidden --moco_type enhanced --moco_gate_hidden 128 --moco_prune_threshold 1e-3 --fusion_strategy self_attention

# Enhanced MoCo（低剪枝阈值）
python main.py --seed=0 --epochs 10 --run_name moco_enhanced_low_prune --moco_type enhanced --moco_gate_hidden 64 --moco_prune_threshold 1e-4 --fusion_strategy self_attention

# Enhanced MoCo（自定义配置）
python main.py --seed=0 --epochs 10 --run_name moco_enhanced_custom --moco_config "enhanced[K=8192,gate_hidden=128,prune_threshold=5e-4]" --fusion_strategy self_attention
```

## 6. 队列大小实验（控制其他变量不变，只改变MoCo队列大小）

```bash
# 小队列
python main.py --seed=0 --epochs 10 --run_name moco_queue_small --fusion_strategy self_attention --moco_type basic --moco_K 1024

# 默认队列（Baseline）
python main.py --seed=0 --epochs 10 --run_name moco_queue_default --fusion_strategy self_attention --moco_type basic --moco_K 4096

# 大队列
python main.py --seed=0 --epochs 10 --run_name moco_queue_large --fusion_strategy self_attention --moco_type basic --moco_K 8192
```