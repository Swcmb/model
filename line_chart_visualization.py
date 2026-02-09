import matplotlib.pyplot as plt
from font_config import setup_chinese_font
setup_chinese_font()  # 自动设置中文字体

import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 实验名称
experiments = [
    "baseline", "self_attention", "co_attention", "hybrid", 
    "hybrid0.5", "tri_co_attention", "tri_self_attention"
]

# 数据值
auroc = [0.9277, 0.9207, 0.9353, 0.9344, 0.9302, 0.9230, 0.9253]
auroc_std = [0.0087, 0.0138, 0.0099, 0.0052, 0.0113, 0.0104, 0.0152]

auprc = [0.9259, 0.9162, 0.9350, 0.9324, 0.9237, 0.9223, 0.9215]
auprc_std = [0.0087, 0.0138, 0.0099, 0.0052, 0.0113, 0.0104, 0.0152]

f1 = [0.8460, 0.8419, 0.8521, 0.8576, 0.8541, 0.8425, 0.8419]
f1_std = [0.0050, 0.0029, 0.0211, 0.0109, 0.0062, 0.0150, 0.0115]

loss = [3.8264, 3.8269, 3.6586, 3.7362, 3.7758, 3.8017, 3.7491]
loss_std = [0.1064, 0.1690, 0.1662, 0.0958, 0.1326, 0.1487, 0.1464]

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 第一个子图：指标折线图
x = np.arange(len(experiments))

# 绘制带误差条的折线图
ax1.errorbar(x, auroc, yerr=auroc_std, marker='o', label='AUROC', capsize=5)
ax1.errorbar(x, auprc, yerr=auprc_std, marker='s', label='AUPRC', capsize=5)
ax1.errorbar(x, f1, yerr=f1_std, marker='^', label='F1', capsize=5)

# 高亮显示最大值
max_auroc_idx = np.argmax(auroc)
max_f1_idx = np.argmax(f1)

ax1.scatter(max_auroc_idx, auroc[max_auroc_idx], color='red', s=100, zorder=5, marker='o')
ax1.annotate(f' {experiments[max_auroc_idx]} AUROC\n{auroc[max_auroc_idx]:.4f}', 
             (max_auroc_idx, auroc[max_auroc_idx]), 
             textcoords="offset points", xytext=(0,10), ha='center')

ax1.scatter(max_f1_idx, f1[max_f1_idx], color='red', s=100, zorder=5, marker='^')
ax1.annotate(f' {experiments[max_f1_idx]} F1\n{f1[max_f1_idx]:.4f}', 
             (max_f1_idx, f1[max_f1_idx]), 
             textcoords="offset points", xytext=(0,10), ha='center')

# 设置第一个子图属性
ax1.set_ylabel('指标值')
ax1.set_title('不同实验策略的指标对比（含标准差）')
ax1.set_xticks(x)
ax1.set_xticklabels(experiments, rotation=45)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# 第二个子图：Loss柱状图
bars = ax2.bar(x, loss, yerr=loss_std, capsize=5, color='lightblue', edgecolor='black', alpha=0.7)

# 用不同的颜色表示误差条
for i, (bar, err) in enumerate(zip(bars, loss_std)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.01, 
             f'{err:.2f}', ha='center', va='bottom', fontsize=9)

# 设置第二个子图属性
ax2.set_ylabel('Loss 值')
ax2.set_title('不同实验策略的 Loss 对比（含标准差）')
ax2.set_xticks(x)
ax2.set_xticklabels(experiments, rotation=45)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局并显示图形
plt.tight_layout()
plt.show()