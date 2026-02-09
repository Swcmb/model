#!/usr/bin/env python3
"""
注意力机制代码结构图生成器
可视化注意力模块的层次结构和关系
"""

def print_attention_hierarchy():
    """打印注意力机制的层次结构"""
    
    print("🏗️  注意力机制架构图")
    print("=" * 80)
    
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                    ATTENTION SYSTEM                      │
    └─────────────────────┬───────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    协作注意力          融合策略          工厂模式
  (Co-Attention)    (Fusion)      (Factory Pattern)
         │                │                │
    ┌────┴────┐     ┌────┴────┐     ┌────┴────┐
    │         │     │         │     │         │
 基础     高级    自注意力   协作注意力  配置解析
 协作     协作    融合     融合       管理
    │         │     │         │     │         │
┌───┴───┐ ┌───┴───┐ ┌───┴───┐ ┌───┴───┐ ┌───┴───┐
│Pairwise│ │Trans- │ │Self   │ │Co-    │ │Config │
│CoAttn │ │former │ │Attn   │ │Attn   │ │Parser │
└───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
    │         │         │         │         │
    │    ┌────┴────┐    │    ┌────┴────┐ │
    │    │ Multi- │    │    │ Hybrid  │ │
    │    │ head   │    │    │ Fusion  │ │
    │    └────┬───┘    │    └────┬───┘ │
    │         │        │         │    │
    │    ┌────┴────┐   │    ┌────┴────┐ │
    │    │Gate     │   │    │Gate     │ │
    │    │Multi-   │   │    │Transformer│
    │    │head     │   │    │Multi-   │ │
    │    └─────────┘   │    │head     │ │
    │                  │    └─────────┘ │
    └──────────────────┴─────────────────┘
""")

def print_class_relationships():
    """打印类之间的关系图"""
    
    print("\n🔗 类关系图")
    print("=" * 80)
    
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                    CLASS RELATIONSHIPS                   │
    └─────────────────────┬───────────────────────────────────┘
                          │
               ┌─────────┴─────────┐
               │                   │
        AttentionFactory      FusionStrategy(ABC)
               │                   │
    ┌──────────┴──────────┐        │
    │                   │        │
 Co-Attention      Fusion    Hybrid Strategy Creation
 Creation          Creation     Methods
    │                   │        │
    │              ┌────┴────┐ │
    │              │         │ │
    │         SelfAttn  CoAttn   │
    │         Fusion     Fusion     │
    │              │         │      │
    │              └────┬────┘      │
    │                   │           │
    │              HybridFusion     │
    │                   │           │
    │                   ▼           ▼
    │         ┌──────────────────────┐
    │         │      EM Model       │
    │         │  (Uses Fusion)     │
    │         └──────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│     CO-ATTENTION CLASSES         │
├─────────────────────────────────────┤
│ • PairwiseCoAttention           │
│ • TransformerPairwiseCoAttention│
│ • MultiHeadPairwiseCoAttention │
│ • GateMultiHeadPairwiseCoAttn   │
└─────────────────────────────────────┘
""")

def print_data_flow():
    """打印数据流图"""
    
    print("\n🌊 数据流图")
    print("=" * 80)
    
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                    DATA FLOW                          │
    └─────────────────────┬───────────────────────────────────┘
                          │
                    输入张量 [B, D]
                          │
                    ┌─────┴─────┐
                    │           │
                实体A      实体B
               [B, D]     [B, D]
                    │           │
                    └─────┬─────┘
                          │
                    协作注意力模块
                          │
                    ┌─────┴─────┐
                    │           │
                A_out      B_out
               [B, D]     [B, D]
                    │           │
                    └─────┬─────┘
                          │
                      特征融合
                          │
                    ┌─────┴─────┐
                    │           │
               拼接选项1     拼接选项2
              [B, 2D]      [B, D]
                    │           │
                    └─────┬─────┘
                          │
                      投影层
                          │
                    ┌─────┴─────┐
                    │           │
                FC Layer1    输出投影
               [B, Hidden]    [B, D]
                    │           │
                    └─────┬─────┘
                          │
                    ┌─────┴─────┐
                    │           │
               FC Layer2    LayerNorm
              [B, 1]       [B, D]
                    │           │
                    └─────┬─────┘
                          │
                      最终输出
                    [B, 1] 或 [B, D]
""")

def print_config_decision_tree():
    """打印配置决策树"""
    
    print("\n🌳 配置决策树")
    print("=" * 80)
    
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                CONFIGURATION DECISION TREE              │
    └─────────────────────┬───────────────────────────────────┘
                          │
                    有高级配置?
                          │
                ┌─────────┴─────────┐
               YES                 NO
                │                 │
            使用配置字符串      智能决策
                │                 │
          ┌─────┴─────┐    ┌─────┴─────┐
          │           │    │           │
    协作注意力   融合策略  多头?      协作注意力?
          │           │    │           │
    ┌─────┴───┐ ┌───┴───┐ YES│      NO┌──────┴──────┐
    │         │ │       │   └───┐    │             │
  基础   Transformer 自注意力   门控?   自注意力    协作注意力
 协作     协作       融合       │     融合        融合
          │           │         │           │             │
          │       ┌───┴───┐   YES│     NO┌─────┴─────┐   │
          │       │       │    └──┐  │     │         │   │
        头数    dropout  多头门控  多头  权重配置   协作类型
        │       │       │    │    │     │         │   │
        ▼       ▼       ▼    ▼    ▼     ▼         ▼   ▼
    [4,8,16] [0.0,0.2] [True] [True] [0.7] [transformer] [pairwise]
""")

def print_performance_matrix():
    """打印性能矩阵"""
    
    print("\n📊 性能特征矩阵")
    print("=" * 80)
    
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │               PERFORMANCE CHARACTERISTICS              │
    └─────────────────────┬───────────────────────────────────┘
                          │
    注意力类型         计算复杂度    内存占用    并行性    推荐场景
    ─────────────────────────────────────────────────────────────
    PairwiseCoAttention
         O(d²)          低          中等       小数据集
    
    TransformerPairwiseCoAttention  
         O(d²+dh)        中          高        中等复杂度
    
    MultiHeadPairwiseCoAttention
         O(d²h)          中          高        高维特征
    
    GateMultiHeadPairwiseCoAttention
         O(d²h+dg)       高          高        精细控制
    
    SelfAttentionFusion
         O(d²)          低          中等       基础融合
    
    HybridFusion
         O(d²+d'²)       中          中等       平衡融合
    
    GateMultiHeadTransformerPairwiseCoAttention
         O(d²h+dg)       高          高        复杂任务
    ─────────────────────────────────────────────────────────────
    注: d=特征维度, h=头数, g=门控参数, d'=融合维度
""")

def main():
    """主函数 - 显示所有结构图"""
    
    print("🎯 注意力机制代码结构分析")
    print("生成时间:", "2025-01-10")
    print("=" * 80)
    
    print_attention_hierarchy()
    print_class_relationships() 
    print_data_flow()
    print_config_decision_tree()
    print_performance_matrix()
    
    print("\n" + "=" * 80)
    print("📁 相关文件")
    print("=" * 80)
    print("""
    📄 layer.py                    # 主要实现文件
    📄 ATTENTION_MODULE_OVERVIEW.md # 详细模块说明
    📄 ATTENTION_CONFIG_USAGE.md   # 配置使用指南
    📄 ATTENTION_REFACTOR_SUMMARY.md # 重构总结
    📄 attention_examples.py      # 配置示例
    📄 test_attention.py          # 功能测试
    📄 parms_setting.py           # 参数配置
    """)
    
    print("\n🚀 快速开始")
    print("=" * 80)
    print("""
    1. 查看所有配置示例:
       python attention_examples.py
    
    2. 测试注意力功能:
       python test_attention.py
    
    3. 运行基础实验:
       python main.py --fusion_strategy self_attention
    
    4. 尝试协作注意力:
       python main.py --use_co_attention --co_attention_type transformer
    
    5. 使用门控多头:
       python main.py --attention_config gated_transformer[num_heads=8,headwise_gate=true]
    """)

if __name__ == "__main__":
    main()