import os
from typing import List, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.calibration import calibration_curve

# 全局绘图风格
sns.set(style="whitegrid", context="talk")

# 中文字体设置：优先使用微软雅黑/黑体/宋体，修复负号显示问题
def _set_chinese_font():
    # 全局高精度绘图参数（提高分辨率与抗锯齿）
    try:
        plt.rcParams.update({
            "savefig.dpi": 300,           # 保存分辨率
            "figure.dpi": 120,            # 交互显示分辨率（适中，避免交互过慢）
            "lines.antialiased": True,    # 抗锯齿
            "patch.antialiased": True,
            "axes.linewidth": 1.2,        # 坐标轴线宽
            "lines.linewidth": 2.0,       # 默认线宽
            "legend.frameon": True,       # 图例带边框
            "legend.framealpha": 0.85,    # 图例透明度
            "pdf.fonttype": 42,           # 兼容性更好的字体嵌入
            "ps.fonttype": 42
        })
    except Exception:
        # 若更新失败，忽略错误，继续字体设置
        pass
    # 中文字体与负号设置
    try:
        # 在 Windows 11 常见可用字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        # 让 seaborn 也使用中文主字体
        try:
            sns.set(font=plt.rcParams['font.sans-serif'][0])
        except Exception:
            pass
    except Exception:
        # 至少保证负号正常显示
        plt.rcParams['axes.unicode_minus'] = False

_set_chinese_font()


def _ensure_dir(path: Optional[str]) -> None:
    """确保保存路径所在目录存在。"""
    if path:
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)


def _finalize(fig: plt.Figure, save_path: Optional[str] = None, dpi: int = 300) -> None:
    """保存或展示图像（增强版：更高 DPI + 自动矢量副本）。
    硬性重定向：若能获取当前 run 的 result_dir，则统一保存到 <run_dir>/figure/<文件名>。
    否则将路径中的 /result/ 替换为 /figure/。
    - 若 save_path 为 PNG/JPG，则额外保存同名 SVG；
    - 若 save_path 为 SVG，则额外保存同名 PNG（用于快速预览）。
    """
    fig.tight_layout()
    if save_path:
        try:
            # 优先通过 get_run_paths 获取当前运行目录
            try:
                from log_output_manager import get_run_paths, make_result_run_dir  # 延迟导入
                paths = get_run_paths() or {}
                run_dir = paths.get("run_result_dir")
                if not run_dir:
                    run_dir = str(make_result_run_dir("data"))
            except Exception:
                run_dir = None

            # 规范化主保存路径
            if run_dir:
                fname = os.path.basename(str(save_path))
                base_target = os.path.join(run_dir, "figure", fname)
            else:
                sp = str(save_path).replace("\\", "/")
                sp = sp.replace("/result/", "/figure/")
                base_target = sp.replace("/", os.sep)

            # 解析扩展名与多格式保存策略
            root, ext = os.path.splitext(base_target)
            ext = ext.lower() if ext else ".png"
            formats = [ext]
            if ext in [".png", ".jpg", ".jpeg"]:
                # 栅格 -> 额外保存矢量
                formats.append(".svg")
            elif ext == ".svg":
                # 矢量 -> 额外保存栅格（便于快速查看）
                formats.append(".png")

            # 依次保存各格式
            for fext in formats:
                target = root + fext
                _ensure_dir(target)
                # 对于矢量（svg/pdf）dpi影响不大，但保持参数统一
                fig.savefig(target, dpi=dpi, bbox_inches="tight")

            plt.close(fig)
        except Exception:
            # 若保存异常，回退为显示
            plt.show()
    else:
        plt.show()


# 1) 损失曲线：支持总损失和分项损失
def plot_loss_curve(
    loss_history: Sequence[float],
    sub_losses: Optional[Dict[str, Sequence[float]]] = None,
    save_path: Optional[str] = None,
    title: str = "训练损失曲线（批次级）"
) -> None:
    """
    - loss_history: 批次级总损失（train.py中有）
    - sub_losses: 可选分项损失（如 {'task_loss': [...], 'cont_loss': [...], 'adv_loss': [...] }）
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, len(loss_history) + 1)
    ax.plot(x, loss_history, label="total_loss", color="#1f77b4", linewidth=2)
    if sub_losses:
        for k, v in sub_losses.items():
            if v is not None and len(v) == len(loss_history):
                ax.plot(x, v, label=k, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("训练批次")
    ax.set_ylabel("损失")
    ax.legend()
    _finalize(fig, save_path)


# 2) 多损失分解：按epoch绘制多条线或堆叠面积
def plot_multi_loss_breakdown(
    epochs: Sequence[int],
    task_loss: Sequence[float],
    cont_loss: Sequence[float],
    adv_loss: Sequence[float],
    stacked: bool = False,
    save_path: Optional[str] = None,
    title: str = "多损失分解（按Epoch）"
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.asarray(epochs)
    if stacked:
        ax.stackplot(x, task_loss, cont_loss, adv_loss, labels=["task_loss", "cont_loss", "adv_loss"], colors=["#1f77b4", "#ff7f0e", "#2ca02c"])
    else:
        ax.plot(x, task_loss, label="task_loss", linewidth=2)
        ax.plot(x, cont_loss, label="cont_loss", linewidth=2)
        ax.plot(x, adv_loss, label="adv_loss", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("损失")
    ax.legend(loc="best")
    _finalize(fig, save_path)


# 3) 训练 vs 验证损失
def plot_train_vs_val_loss(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    save_path: Optional[str] = None,
    title: str = "训练 vs 验证损失（过拟合检查）"
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x_train = np.arange(1, len(train_losses) + 1)
    x_val = np.arange(1, len(val_losses) + 1)
    ax.plot(x_train, train_losses, label="train_loss", color="#1f77b4", linewidth=2)
    ax.plot(x_val, val_losses, label="val_loss", color="#d62728", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("损失")
    ax.legend()
    _finalize(fig, save_path)


# 3.1) 按Epoch绘制：train_loss、val_loss 与 val_AUROC（双y轴）
def _apply_smooth(arr: Sequence[float], method: Optional[str] = None, alpha: float = 0.2, window: int = 3) -> np.ndarray:
    """
    对序列进行可选平滑：
    - method: None | "ema" | "moving"
    - alpha: EMA 平滑系数（0~1）
    - window: 滑动平均窗口大小（>=1）
    """
    x = np.asarray(arr, dtype=np.float64)
    if method is None or len(x) == 0:
        return x
    if method == "ema":
        out = np.empty_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
        return out
    if method == "moving":
        if window <= 1:
            return x
        # 简单滑动平均（居中对齐，边缘用最近值填充）
        kernel = np.ones(window) / float(window)
        y = np.convolve(x, kernel, mode="same")
        # 边缘处理：用原值替换可能的偏差
        y[0] = x[0]
        y[-1] = x[-1]
        return y
    return x


def plot_epoch_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    val_aurocs: Optional[Sequence[float]] = None,
    epochs: Optional[Sequence[int]] = None,
    save_path: Optional[str] = None,
    title: str = "按Epoch的训练/验证损失与验证AUROC曲线",
    smooth: Optional[str] = None,
    smooth_alpha: float = 0.2,
    smooth_window: int = 3,
) -> None:
    """
    - train_losses: 每个Epoch的训练损失
    - val_losses: 每个Epoch的验证损失
    - val_aurocs: 每个Epoch的验证AUROC（可选）
    - epochs: 对应的Epoch索引（可选）。若不提供，则为 1..N
    - smooth: None | "ema" | "moving"
    - smooth_alpha/smooth_window: 平滑参数
    - save_path: 保存路径（可选）
    """
    tl = np.asarray(train_losses, dtype=np.float64)
    vl = np.asarray(val_losses, dtype=np.float64)
    if len(tl) == 0 or len(vl) == 0:
        raise ValueError("train_losses 与 val_losses 不能为空。")
    if len(tl) != len(vl):
        raise ValueError("train_losses 与 val_losses 长度必须一致。")

    n = len(tl)
    if epochs is None:
        x = np.arange(1, n + 1)
    else:
        x = np.asarray(epochs, dtype=np.int64)
        if len(x) != n:
            raise ValueError("epochs 长度需与损失序列一致。")

    # 平滑（默认关闭）
    tl_s = _apply_smooth(tl, method=smooth, alpha=smooth_alpha, window=smooth_window)
    vl_s = _apply_smooth(vl, method=smooth, alpha=smooth_alpha, window=smooth_window)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    # 左轴：损失
    l1, = ax1.plot(x, tl_s, label="train_loss", color="#1f77b4", linewidth=2, marker="o", markersize=3)
    l2, = ax1.plot(x, vl_s, label="val_loss", color="#d62728", linewidth=2, marker="s", markersize=3)
    lines = [l1, l2]
    labels = ["训练损失", "验证损失"]
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("损失")
    ax1.set_title(title)

    # 右轴：AUROC（若提供）
    if val_aurocs is not None:
        va = np.asarray(val_aurocs, dtype=np.float64)
        if len(va) != n:
            raise ValueError("val_aurocs 长度需与损失序列一致。")
        va_s = _apply_smooth(va, method=smooth, alpha=smooth_alpha, window=smooth_window)
        ax2 = ax1.twinx()
        l3, = ax2.plot(x, va_s, label="val_AUROC", color="#2ca02c", linewidth=2, marker="o", markersize=4)
        ax2.set_ylabel("AUROC")
        ax2.set_ylim(-0.02, 1.02)
        lines.append(l3)
        labels.append("验证AUROC")

    # 合并图例到左上角
    ax1.legend(lines, labels, loc="best")
    _finalize(fig, save_path)


def plot_epoch_curves_from_df(
    df: Union[pd.DataFrame, List[Dict]],
    cols: Dict[str, str] = {"epoch": "epoch", "train": "loss_train", "val": "val_loss", "auroc": "val_auroc"},
    save_path: Optional[str] = None,
    title: str = "按Epoch的训练/验证损失与验证AUROC曲线",
    smooth: Optional[str] = None,
    smooth_alpha: float = 0.2,
    smooth_window: int = 3,
) -> None:
    """
    从 DataFrame 绘制曲线。默认列名（val 为必需，auroc 可选）：
    - epoch: 'epoch'
    - train: 'loss_train'（训练保存CSV中使用的列名）
    - val: 'val_loss'
    - auroc: 'val_auroc'（若无则不绘制AUROC）
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # 读取 epoch
    epoch_col = cols.get("epoch", "epoch")
    if epoch_col not in df.columns:
        raise ValueError(f"DataFrame 缺少列：{epoch_col}")
    epochs = df[epoch_col].to_numpy()

    # 读取训练损失
    train_col = cols.get("train", "loss_train")
    if train_col not in df.columns:
        raise ValueError(f"DataFrame 缺少训练损失列：{train_col}")
    train_losses = df[train_col].to_numpy()

    # 读取验证损失（必需）
    val_col = cols.get("val", "val_loss")
    if val_col not in df.columns:
        raise ValueError(f"DataFrame 缺少必需列：{val_col}")
    val_losses = df[val_col].to_numpy()

    # 读取 AUROC（可缺省）
    auroc_col = cols.get("auroc", "val_auroc")
    val_aurocs = None
    if auroc_col and (auroc_col in df.columns):
        val_aurocs = df[auroc_col].to_numpy()

    plot_epoch_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        val_aurocs=val_aurocs,
        epochs=epochs,
        save_path=save_path,
        title=title,
        smooth=smooth,
        smooth_alpha=smooth_alpha,
        smooth_window=smooth_window,
    )


# 4) 学习率调度
def plot_lr_schedule(
    lrs: Sequence[float],
    save_path: Optional[str] = None,
    title: str = "学习率调度曲线"
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(1, len(lrs) + 1)
    ax.plot(x, lrs, color="#9467bd", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("学习率")
    _finalize(fig, save_path)


# 5) 每Epoch指标柱状图（AUROC/AUPRC/F1）
def plot_epoch_metrics_bar(
    epoch_metrics: Union[pd.DataFrame, List[Dict]],
    metrics: List[str] = ["auroc", "auprc", "f1"],
    save_path: Optional[str] = None,
    title: str = "Epoch 指标汇总（柱状）"
) -> None:
    if not isinstance(epoch_metrics, pd.DataFrame):
        epoch_metrics = pd.DataFrame(epoch_metrics)
    fig, ax = plt.subplots(figsize=(12, 5))
    df = epoch_metrics[["epoch"] + metrics].melt(id_vars="epoch", var_name="metric", value_name="value")
    sns.barplot(data=df, x="epoch", y="value", hue="metric", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("指标值")
    ax.legend(loc="best")
    _finalize(fig, save_path)


# 6) ROC 曲线
def plot_roc_curve(
    y_true: Sequence[int],
    y_score: Sequence[float],
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC (AUROC={auc:.4f})", color="#1f77b4", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title(title or "ROC 曲线")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    _finalize(fig, save_path)


# 7) PR 曲线
def plot_pr_curve(
    y_true: Sequence[int],
    y_score: Sequence[float],
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label=f"PR (AUPRC={ap:.4f})", color="#ff7f0e", linewidth=2)
    ax.set_title(title or "Precision-Recall 曲线")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="best")
    _finalize(fig, save_path)


# 8) 校准曲线（预测概率 vs. 真实分数）
def plot_calibration_curve(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
    save_path: Optional[str] = None,
    title: str = "概率校准曲线"
) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, "s-", label="校准", color="#2ca02c")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("预测概率分箱均值")
    ax.set_ylabel("真实阳性率")
    ax.legend(loc="best")
    _finalize(fig, save_path)


# 9) 阈值扫描图（F1 vs. Threshold）
def plot_threshold_scan(
    thresholds: Sequence[float],
    f1_vals: Sequence[float],
    save_path: Optional[str] = None,
    title: str = "F1 vs. 阈值扫描"
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1_vals, color="#d62728", linewidth=2)
    best_idx = int(np.argmax(f1_vals)) if len(f1_vals) > 0 else None
    if best_idx is not None:
        ax.axvline(thresholds[best_idx], color="#d62728", linestyle="--", alpha=0.6, label=f"best={thresholds[best_idx]:.3f}, F1={f1_vals[best_idx]:.4f}")
    ax.set_title(title)
    ax.set_xlabel("阈值")
    ax.set_ylabel("F1")
    ax.legend(loc="best")
    _finalize(fig, save_path)


# 10) 温度缩放效果（可靠性图 + ECE 前后对比）
def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """计算期望校准误差 ECE。"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        l, r = bins[i], bins[i + 1]
        mask = (y_prob >= l) & (y_prob < r)
        if mask.sum() == 0:
            continue
        bin_acc = (y_true[mask] == 1).mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def plot_temperature_scaling_effect(
    y_true: Sequence[int],
    logits: Sequence[float],
    T_opt: Optional[float],
    n_bins: int = 10,
    save_path: Optional[str] = None,
    title: str = "温度缩放效果（可靠性/ECE）"
) -> None:
    """
    - y_true: 真实标签
    - logits: 未Sigmoid的原始logits
    - T_opt: 最优温度（train.py/test()中已网格搜索 best_T）
    """
    y_true_np = np.asarray(y_true, dtype=np.int64)
    logits_np = np.asarray(logits, dtype=np.float32)
    probs_before = 1.0 / (1.0 + np.exp(-logits_np))
    if T_opt is not None:
        probs_after = 1.0 / (1.0 + np.exp(-logits_np / float(T_opt)))
    else:
        probs_after = probs_before.copy()

    ece_before = _compute_ece(y_true_np, probs_before, n_bins=n_bins)
    ece_after = _compute_ece(y_true_np, probs_after, n_bins=n_bins)

    # 可靠性图
    fig, ax = plt.subplots(figsize=(6, 6))
    bt, bp = calibration_curve(y_true_np, probs_before, n_bins=n_bins, strategy="uniform")
    at, ap = calibration_curve(y_true_np, probs_after, n_bins=n_bins, strategy="uniform")
    ax.plot(bp, bt, "o-", label=f"校准前 (ECE={ece_before:.4f})", color="#7f7f7f")
    ax.plot(ap, at, "s-", label=f"校准后 (ECE={ece_after:.4f}, T={T_opt})", color="#1f77b4")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("预测概率分箱均值")
    ax.set_ylabel("真实阳性率")
    ax.legend(loc="best")
    _finalize(fig, save_path)


# 11) 每折性能比较（箱线或小提琴）
def plot_per_fold_comparison(
    fold_results: List[Dict[str, float]],
    use_violin: bool = False,
    metrics: List[str] = ["auroc", "auprc", "f1"],
    save_path: Optional[str] = None,
    title: str = "5折性能比较"
) -> None:
    df = pd.DataFrame(fold_results)
    df = df[metrics]
    df_melt = df.melt(var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(8, 6))
    if use_violin:
        sns.violinplot(data=df_melt, x="metric", y="value", inner="box", ax=ax)
    else:
        sns.boxplot(data=df_melt, x="metric", y="value", ax=ax)
    sns.stripplot(data=df_melt, x="metric", y="value", color="black", size=4, alpha=0.6, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("指标")
    ax.set_ylabel("值")
    _finalize(fig, save_path)


# 12) 混淆矩阵热力图
def plot_confusion_matrix_heatmap(
    cm: Union[Tuple[int, int, int, int], np.ndarray],
    normalize: bool = False,
    save_path: Optional[str] = None,
    title: str = "混淆矩阵热力图"
) -> None:
    """
    - cm: 可传 (tn, fp, fn, tp) 或 2x2矩阵
    """
    if isinstance(cm, tuple) or isinstance(cm, list):
        tn, fp, fn, tp = cm
        mat = np.array([[tn, fp], [fn, tp]], dtype=np.float64)
    else:
        mat = np.asarray(cm, dtype=np.float64)
        if mat.shape != (2, 2):
            raise ValueError("混淆矩阵必须为2x2或(tn, fp, fn, tp)。")

    disp = mat.copy()
    if normalize:
        row_sum = disp.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        disp = disp / row_sum

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(disp, annot=True, fmt=".3f" if normalize else "g", cmap="Blues", cbar=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_xticklabels(["负类", "正类"])
    ax.set_yticklabels(["负类", "正类"])
    _finalize(fig, save_path)


# ==================== 辅助加载器（可选） ====================

def load_epoch_metrics_csv(csv_path: str) -> pd.DataFrame:
    """
    读取 train.py 保存的 metrics/train_epoch_metrics_*.csv 文件。
    返回包含 epoch、loss_train、task_loss、cont_loss、adv_loss、auroc、auprc、precision、recall、f1、tn、fp、fn、tp 的 DataFrame。
    """
    df = pd.read_csv(csv_path)
    # 保证类型正确
    for col in ["epoch", "tn", "fp", "fn", "tp"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def derive_threshold_scan_arrays(txt_path: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    从 threshold_scan_*.txt 中解析 best_threshold/best_temperature/calibrated_best_threshold/F1。
    返回：(best_t, best_f1, best_t_cal, best_f1_cal)，可能为 None。
    """
    best_t = best_f1 = best_t_cal = best_f1_cal = None
    if not os.path.exists(txt_path):
        return best_t, best_f1, best_t_cal, best_f1_cal
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("best_threshold"):
                try:
                    parts = s.split()
                    best_t = float(parts[0].split("=")[1])
                    best_f1 = float(parts[1].split("=")[1])
                except Exception:
                    pass
            elif s.startswith("best_temperature"):
                # 仅用于展示，不在此函数返回
                pass
            elif s.startswith("calibrated_best_threshold"):
                try:
                    parts = s.split()
                    best_t_cal = float(parts[0].split("=")[1])
                    best_f1_cal = float(parts[1].split("=")[1])
                except Exception:
                    pass
    return best_t, best_f1, best_t_cal, best_f1_cal


# ==================== 使用示例（供参考，非运行入口） ====================
# 训练完成后，你可以：
# df = load_epoch_metrics_csv("EM/result/.../metrics/train_epoch_metrics_fold_1_XXXX.csv")
# plot_multi_loss_breakdown(df["epoch"], df["task_loss"], df["cont_loss"], df["adv_loss"], stacked=False, save_path="OUTPUT/result/loss_breakdown.png")
# plot_epoch_metrics_bar(df, metrics=["auroc","auprc","f1"], save_path="OUTPUT/result/epoch_metrics_bar.png")
# 对测试阶段：
# plot_roc_curve(y_true, y_score, save_path="OUTPUT/result/roc.png")
# plot_pr_curve(y_true, y_score, save_path="OUTPUT/result/pr.png")
# plot_calibration_curve(y_true, y_prob, save_path="OUTPUT/result/calibration.png")
# plot_threshold_scan(ths, f1_vals, save_path="OUTPUT/result/threshold_scan.png")
# plot_temperature_scaling_effect(y_true, logits, T_opt, save_path="OUTPUT/result/temperature_effect.png")
# plot_per_fold_comparison(all_fold_results, use_violin=False, save_path="OUTPUT/result/per_fold_box.png")
# plot_confusion_matrix_heatmap((tn,fp,fn,tp), normalize=False, save_path="OUTPUT/result/cm.png")