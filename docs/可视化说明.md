# 可视化说明（Visualization README）

本仓库提供了一组标准化的训练/验证阶段可视化函数与输出图，帮助快速诊断模型训练状态、过拟合、指标变化与阈值选择等问题。本文档说明各张图“是做什么的、怎么看、如何复现”。

建议先阅读保存规则与数据来源，再按需要查看具体图种类。

---

## 一、图像保存规则与目录规范

- 统一保存目录
  - 训练/评估期间，图像最终保存到当前运行目录下的 `<run_result_dir>/figure/` 中（例如：`OUTPUT/result/LDA_C_r1_005_data_20251020_155500/figure/`）。
  - 函数内部会尝试通过 `log_output_manager.get_run_paths()` 定位到本次运行目录；若不可用，则会将用户传入的路径中的 `/result/` 替换为 `/figure/` 保存。
- 多格式自动保存
  - 若保存为 PNG/JPG，会额外保存同名 SVG 作为矢量图；若保存为 SVG，会额外保存同名 PNG 便于快速预览。
- 函数位置
  - 主要位于 `model/visualization.py`，并包含数据加载辅助函数。

---

## 二、数据来源（metrics）

大多数按 epoch 绘图依赖 `metrics/` 子目录的 CSV/JSON：
- 每折训练 epoch 指标（CSV）
  - `metrics/train_epoch_metrics_fold_{k}_{run_id}.csv`
  - 典型列：`epoch, loss_train, task_loss, cont_loss, adv_loss, auroc, auprc, precision, recall, f1, tn, fp, fn, tp`
- 阈值扫描与温度缩放
  - `metrics/threshold_scan_fold_{k}_{run_id}.csv`
  - `metrics/threshold_scan_calibrated_fold_{k}_{run_id}.csv`
  - `metrics/temperature_fold_{k}_{run_id}.json`（温度参数）
- 预测对与真值
  - `metrics/y_true_pred_fold_{k}_{run_id}.csv`（便于外部重绘 ROC/PR）

示例所在目录（已存在的输出，可直接参考同名图）：  
`OUTPUT/result/LDA_C_r1_005_data_20251020_155500/`

---

## 三、各类图与判读要点

### 1) 每折 Epoch 曲线：train_loss / val_loss / val_AUROC
- 文件名（每折一张）：`epoch_curves_fold_{k}.png`
- 对应函数：
  - `plot_epoch_curves(train_losses, val_losses, val_aurocs, epochs, save_path=...)`
  - `plot_epoch_curves_from_df(df, cols=..., save_path=...)`
- 坐标轴：
  - 左轴：`train_loss`（蓝）、`val_loss`（红）
  - 右轴：`val_AUROC`（绿）
- 用途与判读：
  - 过拟合：`train_loss` 持续下降，但 `val_loss` 出现明显回升；`val_AUROC` 不升反降或停滞。
  - 欠拟合/振荡：两条损失均高且波动较大，`val_AUROC` 无上升趋势或上下振荡。
- 常见位置：`OUTPUT/result/<run>/figure/epoch_curves_fold_1.png ... fold_5.png`

### 2) 多损失分解（按 Epoch）
- 文件名（每折一张）：`loss_breakdown_fold_{k}.png`
- 对应函数：`plot_multi_loss_breakdown(epochs, task_loss, cont_loss, adv_loss, stacked=False, ...)`
- 用途与判读：
  - 观测任务损失、对比损失、对抗损失三者的相对规模与收敛特征。
  - `stacked=True` 可看总量构成；`stacked=False` 看各项曲线趋势。

### 3) 每 Epoch 指标柱状图（AUROC/AUPRC/F1）
- 文件名（每折一张）：`epoch_metrics_bar_fold_{k}.png`
- 对应函数：`plot_epoch_metrics_bar(epoch_metrics_df, metrics=["auroc","auprc","f1"], ...)`
- 用途与判读：
  - 快速比较不同 epoch 的核心指标水平；选择“早停/最佳 epoch”可参考峰值所在。

### 4) ROC 曲线
- 文件名（每折一张）：`roc_fold_{k}.png`
- 对应函数：`plot_roc_curve(y_true, y_score, ...)`
- 用途与判读：
  - 观察在不同阈值下的 TPR-FPR 权衡；图下的 AUROC 数值越高越好。

### 5) PR 曲线
- 文件名（每折一张）：`pr_fold_{k}.png`
- 对应函数：`plot_pr_curve(y_true, y_score, ...)`
- 用途与判读：
  - 类别不均衡时更有意义；关注 AUPRC（面积越大越好）。

### 6) 概率校准曲线（含温度缩放效果）
- 文件名（每折一张）：
  - 校准曲线：`calibration_fold_{k}.png`
  - 温度缩放对比：`temperature_effect_fold_{k}.png`
- 对应函数：
  - `plot_calibration_curve(y_true, y_prob, ...)`
  - `plot_temperature_scaling_effect(y_true, logits, T_opt, ...)`
- 用途与判读：
  - 预测概率是否“可信”（与实际频率匹配）。温度缩放后理想曲线应更贴近对角线（ECE 下降）。

### 7) 阈值扫描（含校准后）
- 文件名（每折一张）：
  - 原始：`threshold_scan_fold_{k}.png`
  - 校准后：`threshold_scan_calibrated_fold_{k}.png`
- 对应函数：`plot_threshold_scan(thresholds, f1_vals, ...)`
- 用途与判读：
  - 搜索最优分类阈值（如 F1 最大处）；对比校准前后最优阈值与 F1 的变化。

### 8) 混淆矩阵热力图
- 文件名（汇总/或每折分别）：`confusion_matrix_sum.png`、`confusion_matrix_fold_{k}.png`（如有）
- 对应函数：`plot_confusion_matrix_heatmap(cm, normalize=False, ...)`
- 用途与判读：
  - 查看 TP/FP/FN/TN 绝对量或归一化比例；失衡方向与错误类型一目了然。

### 9) 跨折性能比较（箱线/小提琴）
- 文件名：`per_fold_box.png`
- 对应函数：`plot_per_fold_comparison(fold_results, use_violin=False, metrics=["auroc","auprc","f1"], ...)`
- 用途与判读：
  - 对比各折分布、稳定性与离群情况；选择箱线或小提琴以展示分布差异。

### 10) 学习率调度曲线
- 文件名（如保存）：`lr_schedule.png`
- 对应函数：`plot_lr_schedule(lrs, ...)`
- 用途与判读：
  - 可视化学习率随 epoch 的变化策略（Step/Cosine/Warmup 等）。

---

## 四、如何复现这些图（示例）

以每折 epoch 曲线为例（从 CSV 读取）：
```python
from model.visualization import load_epoch_metrics_csv, plot_epoch_curves_from_df

# 选择某一次运行和某折
run_dir = "OUTPUT/result/LDA_C_r1_005_data_20251020_155500"
csv_path = f"{run_dir}/metrics/train_epoch_metrics_fold_1_20251020_155500.csv"

df = load_epoch_metrics_csv(csv_path)

# 绘制并自动保存到本次运行的 figure/ 目录中（内部会同时存 PNG/SVG）
plot_epoch_curves_from_df(
    df,
    cols={"epoch":"epoch", "train":"loss_train", "val":"val_loss", "auroc":"auroc"},
    save_path=f"{run_dir}/figure/epoch_curves_fold_1.png",
    title="Fold-1: 训练/验证损失与验证AUROC"
)
```

多损失分解：
```python
from model.visualization import load_epoch_metrics_csv, plot_multi_loss_breakdown

df = load_epoch_metrics_csv(csv_path)
plot_multi_loss_breakdown(
    epochs=df["epoch"],
    task_loss=df["task_loss"],
    cont_loss=df["cont_loss"],
    adv_loss=df["adv_loss"],
    stacked=False,
    save_path=f"{run_dir}/figure/loss_breakdown_fold_1.png",
    title="Fold-1: 多损失分解（按Epoch）"
)
```

阈值扫描（F1 vs 阈值）：
```python
import pandas as pd
from model.visualization import plot_threshold_scan

scan_csv = f"{run_dir}/metrics/threshold_scan_fold_1_20251020_155500.csv"
scan = pd.read_csv(scan_csv)
plot_threshold_scan(
    thresholds=scan["threshold"],
    f1_vals=scan["f1"],
    save_path=f"{run_dir}/figure/threshold_scan_fold_1.png",
    title="Fold-1: F1 vs 阈值"
)
```

---

## 五、过拟合/欠拟合的快速判断（基于图像）

优先看 “每折 Epoch 曲线（epoch_curves_fold_*.png）”：
- 过拟合：`train_loss` 继续下降，`val_loss` 开始上升；`val_AUROC` 不再提升甚至下降。
- 欠拟合/振荡：`train_loss` 与 `val_loss` 都较高、波动大，`val_AUROC` 没明显上升趋势。
- 稳定/收敛良好：`train_loss` 与 `val_loss` 逐步下降并收敛，`val_AUROC` 稳步提升或持平在高值。

（可选）后续可扩展在图上自动加“过拟合/欠拟合/稳定”标签与最优 epoch 标记。

---

## 六、自动标注（过拟合/欠拟合/最佳 epoch）

无需改动现有绘图函数，你可以用下面的简易脚本基于 CSV 直接生成“带标注”的图（判定基于最近 `window` 个 epoch 的线性趋势）：

判定启发式（可按需调整）：
- 过拟合：最近窗口内 `slope(train_loss) < -eps` 且 `slope(val_loss) > eps`
- 欠拟合/振荡：`|slope(train_loss)| <= eps` 且 `|slope(val_loss)| <= eps` 且 `val_AUROC` 最近窗口提升 < `delta`
- 否则：稳定/收敛中
- 最佳 epoch：默认取 `val_loss` 最小；如果设置 `best_by="auroc"` 则取 `val_AUROC` 最大

完整示例：
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 参数
run_dir = "OUTPUT/result/LDA_C_r1_005_data_20251020_155500"
fold = 1
best_by = "val_loss"  # 可选 "val_loss" 或 "auroc"
window = 5
eps = 1e-3
delta = 1e-3

df = pd.read_csv(f"{run_dir}/metrics/train_epoch_metrics_fold_{fold}_20251020_155500.csv")
epochs = df["epoch"].to_numpy()
train = df["loss_train"].to_numpy()
val = df["val_loss"].to_numpy() if "val_loss" in df.columns else None
auroc = df["auroc"].to_numpy() if "auroc" in df.columns else None

# 选择最近窗口
def lin_slope(y):
    if len(y) < 2: return 0.0
    x = np.arange(len(y))
    k, b = np.polyfit(x, y, 1)
    return float(k)

w = min(window, len(train))
sl_train = lin_slope(train[-w:])
sl_val = lin_slope(val[-w:]) if val is not None else 0.0
au_delta = float(auroc[-1] - auroc[-w]) if (auroc is not None and len(auroc) >= w) else 0.0

# 判定
if (sl_train < -eps) and (sl_val > eps):
    status = "过拟合"
elif (abs(sl_train) <= eps) and (abs(sl_val) <= eps) and (abs(au_delta) < delta):
    status = "欠拟合/振荡"
else:
    status = "稳定/收敛中"

# 最佳 epoch
if best_by == "auroc" and auroc is not None:
    best_idx = int(np.nanargmax(auroc))
    best_label = f"best AUROC={auroc[best_idx]:.4f}"
else:
    # 默认 val_loss 最小
    best_idx = int(np.nanargmin(val)) if val is not None else int(np.nanargmin(train))
    best_label = f"best val_loss={val[best_idx]:.4f}" if val is not None else f"best train_loss={train[best_idx]:.4f}"
best_epoch = int(epochs[best_idx])

# 画图 + 标注
fig, ax1 = plt.subplots(figsize=(11,5))
ax1.plot(epochs, train, label="train_loss", color="#1f77b4", marker="o", ms=3)
if val is not None:
    ax1.plot(epochs, val, label="val_loss", color="#d62728", marker="s", ms=3)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("损失")

ax2 = ax1.twinx()
if auroc is not None:
    ax2.plot(epochs, auroc, label="val_AUROC", color="#2ca02c", marker="o", ms=4)
    ax2.set_ylabel("AUROC"); ax2.set_ylim(-0.02, 1.02)

# 最佳 epoch 竖线与文字
ax1.axvline(best_epoch, color="#9467bd", ls="--", lw=1.5, alpha=0.8)
ax1.text(best_epoch, ax1.get_ylim()[1]*0.95, f"最佳 epoch={best_epoch}\n{best_label}",
         color="#9467bd", ha="center", va="top", fontsize=10, bbox=dict(fc="white", ec="#9467bd", alpha=0.7))

# 训练状态标注
ax1.text(0.02, 0.98, f"训练状态：{status}\n"
         f"slope(train)={sl_train:+.3e}\n"
         f"slope(val)={sl_val:+.3e}\n"
         f"ΔAUROC(last{w})={au_delta:+.3e}",
         transform=ax1.transAxes, ha="left", va="top",
         fontsize=10, bbox=dict(fc="white", ec="#333", alpha=0.8))

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels() if auroc is not None else ([],[])
ax1.legend(lines1+lines2, labels1+labels2, loc="best")

ax1.set_title(f"Fold-{fold}: 训练/验证损失与验证AUROC（自动标注）")
plt.tight_layout()
plt.savefig(f"{run_dir}/figure/epoch_curves_annotated_fold_{fold}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{run_dir}/figure/epoch_curves_annotated_fold_{fold}.svg", dpi=300, bbox_inches="tight")
plt.close(fig)
```

批量处理 5 折（示意）：
```python
for k in range(1, 6):
    # 将上段脚本中的 fold 替换为 k，循环生成 epoch_curves_annotated_fold_k.{png,svg}
    pass
```

> 注：
> - `window/eps/delta` 可根据曲线噪声调节；窗口越大，趋势越稳。
> - 若训练早停，最佳 epoch 通常来自验证指标（建议使用 `best_by="auroc"`）；若仅关注损失，使用 `val_loss` 最小。
> - 需要平滑时，可在计算斜率前对 `train/val/auroc` 做 EMA 或滑动平均（例如 `pd.Series(train).ewm(alpha=0.2).mean()`）。

---

## 七、函数索引（visualization.py）

- 损失与指标类
  - `plot_loss_curve(...)`（批次级总损失与分项）
  - `plot_multi_loss_breakdown(...)`（按 epoch 的任务/对比/对抗损失）
  - `plot_train_vs_val_loss(...)`（基础版训练 vs 验证损失）
  - `plot_epoch_curves(...)`、`plot_epoch_curves_from_df(...)`（核心：train/val loss + val AUROC）
  - `plot_epoch_metrics_bar(...)`（AUROC/AUPRC/F1 柱状）
  - `plot_lr_schedule(...)`（学习率）
- ROC/PR/校准/阈值/混淆矩阵/跨折
  - `plot_roc_curve(...)`
  - `plot_pr_curve(...)`
  - `plot_calibration_curve(...)`
  - `plot_temperature_scaling_effect(...)`
  - `plot_threshold_scan(...)`
  - `plot_confusion矩阵热力图(...)`
  - `plot_per_fold_comparison(...)`
- 数据加载与解析
  - `load_epoch_metrics_csv(csv_path)`
  - `derive_threshold_scan_arrays(txt_path)`

---

## 八、常见问答

- Q: “过拟合该看哪张图？”
  - A: 看 `epoch_curves_fold_{k}.png`。这张同时展示 train_loss、val_loss 与 val_AUROC，最能直观体现过拟合。
- Q: “为什么有时同名文件多出一种格式？”
  - A: 为兼容与排版需要，PNG/JPG 会附带 SVG 矢量版本，SVG 会附带 PNG 预览版本。
- Q: “CSV 列名不一致怎么办？”
  - A: `plot_epoch_curves_from_df` 支持通过 `cols={...}` 指定列映射，如 `{"train":"loss_train", "val":"val_loss"}`。

---

如需把“自动标注”内的逻辑固化到 `plot_epoch_curves(...)` 内部作为可选参数（例如 `annotate=True`），我可以在修复 `visualization.py` 顶部语法错误后，安全地加入该特性并保证与现有输出兼容。