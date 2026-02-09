import matplotlib.pyplot as plt
from font_config import setup_chinese_font
setup_chinese_font()  # 自动设置中文字体

import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_baseline_vs_view_ablation():
    """
    绘制基线模型与视图消融实验对比图
    """
    # 数据
    methods = ['Baseline', 'Disable View 0']
    auroc = [0.9277, 0.9303]
    auroc_std = [0.0087, 0.0118]
    auprc = [0.9259, 0.9307]
    auprc_std = [0.0087, 0.0118]
    f1 = [0.8460, 0.8402]
    f1_std = [0.0050, 0.0210]
    loss = [3.826, 3.801]
    loss_std = [0.106, 0.149]  # 估算值
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('基线模型与视图消融 (Baseline vs. View Ablation)', fontsize=16)
    
    # AUROC对比
    bars1 = axes[0, 0].bar(methods, auroc, yerr=auroc_std, capsize=5, color=['skyblue', 'lightcoral'])
    axes[0, 0].set_title('AUROC对比')
    axes[0, 0].set_ylabel('AUROC')
    
    # AUPRC对比
    bars2 = axes[0, 1].bar(methods, auprc, yerr=auprc_std, capsize=5, color=['skyblue', 'lightcoral'])
    axes[0, 1].set_title('AUPRC对比')
    axes[0, 1].set_ylabel('AUPRC')
    
    # F1-Score对比
    bars3 = axes[1, 0].bar(methods, f1, yerr=f1_std, capsize=5, color=['skyblue', 'lightcoral'])
    axes[1, 0].set_title('F1-Score对比')
    axes[1, 0].set_ylabel('F1-Score')
    
    # Loss对比
    bars4 = axes[1, 1].bar(methods, loss, yerr=loss_std, capsize=5, color=['skyblue', 'lightcoral'])
    axes[1, 1].set_title('Loss对比')
    axes[1, 1].set_ylabel('Loss')
    
    # 添加数值标签
    for ax, values, stds in zip([axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]], 
                                [auroc, auprc, f1, loss], 
                                [auroc_std, auprc_std, f1_std, loss_std]):
        for i, (v, s) in enumerate(zip(values, stds)):
            ax.text(i, v + s + 0.001, f'{v:.4f}±{s:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('baseline_vs_view_ablation.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_fusion_strategies():
    """
    绘制融合策略对比图
    """
    # 数据
    strategies = ['Self-Attention', 'Co-Attention', 'Hybrid\n(Default)', 'Hybrid\n(w=0.5)']
    auroc = [0.9207, 0.9353, 0.9344, 0.9302]
    auprc = [0.9162, 0.9350, 0.9324, 0.9237]
    f1 = [0.8419, 0.8521, 0.8576, 0.8541]
    loss = [3.826, 3.658, 3.736, 3.775]
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('融合策略对比 (Fusion Strategies)', fontsize=16)
    
    # AUROC对比
    bars1 = axes[0, 0].bar(strategies, auroc, color=['gold', 'limegreen', 'deepskyblue', 'orange'])
    axes[0, 0].set_title('AUROC对比')
    axes[0, 0].set_ylabel('AUROC')
    
    # AUPRC对比
    bars2 = axes[0, 1].bar(strategies, auprc, color=['gold', 'limegreen', 'deepskyblue', 'orange'])
    axes[0, 1].set_title('AUPRC对比')
    axes[0, 1].set_ylabel('AUPRC')
    
    # F1-Score对比
    bars3 = axes[1, 0].bar(strategies, f1, color=['gold', 'limegreen', 'deepskyblue', 'orange'])
    axes[1, 0].set_title('F1-Score对比')
    axes[1, 0].set_ylabel('F1-Score')
    
    # Loss对比
    bars4 = axes[1, 1].bar(strategies, loss, color=['gold', 'limegreen', 'deepskyblue', 'orange'])
    axes[1, 1].set_title('Loss对比')
    axes[1, 1].set_ylabel('Loss')
    
    # 添加数值标签
    for ax, values in zip([axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]], 
                          [auroc, auprc, f1, loss]):
        for i, v in enumerate(values):
            ax.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('fusion_strategies.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_tri_entity_mode():
    """
    绘制三元实体模式探索图
    """
    # 数据
    configs = ['Standard\nCo-Attention', 'Tri +\nCo-Attention', 'Tri +\nSelf-Attention']
    auroc = [0.9353, 0.9230, 0.9253]
    auprc = [0.9350, 0.9223, 0.9215]
    f1 = [0.8521, 0.8425, 0.8419]
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('三元实体模式探索 (Tri-Entity Mode)', fontsize=16)
    
    # AUROC对比
    bars1 = axes[0].bar(configs, auroc, color=['limegreen', 'gold', 'orange'])
    axes[0].set_title('AUROC对比')
    axes[0].set_ylabel('AUROC')
    
    # AUPRC对比
    bars2 = axes[1].bar(configs, auprc, color=['limegreen', 'gold', 'orange'])
    axes[1].set_title('AUPRC对比')
    axes[1].set_ylabel('AUPRC')
    
    # F1-Score对比
    bars3 = axes[2].bar(configs, f1, color=['limegreen', 'gold', 'orange'])
    axes[2].set_title('F1-Score对比')
    axes[2].set_ylabel('F1-Score')
    
    # 添加数值标签
    for ax, values in zip(axes, [auroc, auprc, f1]):
        for i, v in enumerate(values):
            ax.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('tri_entity_mode.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_summary_dashboard():
    """
    绘制整体性能汇总图
    """
    # 数据
    models = ['Co-Attention', 'Hybrid', 'Disable\nView 0', 'Baseline', 'Tri-Entity\nModes']
    auroc = [0.9353, 0.9344, 0.9303, 0.9277, 0.9240]  # 近似Tri-Entity值
    f1 = [0.8521, 0.8576, 0.8402, 0.8460, 0.8420]    # 近似Tri-Entity值
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(models))
    width = 0.35
    
    # 绘制柱状图
    bars1 = ax.bar(x - width/2, auroc, width, label='AUROC', color='skyblue')
    bars2 = ax.bar(x + width/2, f1, width, label='F1-Score', color='lightcoral')
    
    # 添加排名标记
    rankings = ['1', '2', '3', '4', '5']
    for i, rank in enumerate(rankings):
        ax.text(i, max(auroc[i], f1[i]) + 0.01, rank, ha='center', va='bottom', fontsize=16)
    
    # 添加数值标签
    for i, (a, f) in enumerate(zip(auroc, f1)):
        ax.text(i - width/2, a + 0.005, f'{a:.4f}', ha='center', va='bottom')
        ax.text(i + width/2, f + 0.005, f'{f:.4f}', ha='center', va='bottom')
    
    ax.set_xlabel('模型/策略')
    ax.set_ylabel('指标值')
    ax.set_title('整体性能汇总 (Summary Dashboard)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_baseline_vs_view_ablation()
    plot_fusion_strategies()
    plot_tri_entity_mode()
    plot_summary_dashboard()