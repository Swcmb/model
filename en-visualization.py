import os
from typing import List, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.calibration import calibration_curve

# 定义公共接口
__all__ = [
    'roc_curve', 'roc_auc_score', 'precision_recall_curve', 
    'average_precision_score', 'confusion_matrix', 'calibration_curve'
]

# 全局绘图风格
sns.set(style="whitegrid", context="talk")

# 中文字体设置：优先使用微软雅黑/黑体/宋体，修复负号显示问题
def _set_chinese_font():
    """
    设置中文字体和全局绘图参数
    
    该函数配置matplotlib的中文字体支持和全局绘图参数，包括:
    1. 设置中文字体族，优先使用微软雅黑、黑体、宋体等常见中文字体
    2. 配置负号显示
    3. 设置高分辨率和抗锯齿参数以提升图表质量
    """
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
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS','DejaVu Sans Mono']
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
    """
    保存或展示图像（增强版：更高 DPI + 自动矢量副本）。
    
    该函数负责处理图表的最终输出，支持自动保存多种格式和规范路径管理:
    硬性重定向：若能获取当前 run 的 result_dir，则统一保存到 <run_dir>/figure/<文件名>。
    否则将路径中的 /result/ 替换为 /figure/。
    - 若 save_path 为 PNG/JPG，则额外保存同名 SVG；
    - 若 save_path 为 SVG，则额外保存同名 PNG（用于快速预览）。
    
    参数:
        fig: matplotlib图形对象
        save_path: 保存路径，如果为None则直接显示图表
        dpi: 保存图像的分辨率，默认为300
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


# 1) losscurve：支持总loss和分项loss
def plot_loss_curve(
    loss_history: Sequence[float],
    sub_losses: Optional[Dict[str, Sequence[float]]] = None,
    save_path: Optional[str] = None,
    title: str = "Training Loss Curve (Batch Level)"
) -> None:
    """
    Plot training loss curve, supporting total loss and sub-item losses

    Args:
        loss_history (Sequence[float]): Batch-level total loss sequence
        sub_losses (Optional[Dict[str, Sequence[float]]]): Optional sub-loss dictionary, keys are loss names,
                                                         values are corresponding loss sequences
                                                         e.g. {'task_loss': [...], 'cont_loss': [...], 'adv_loss': [...]}
        save_path (Optional[str]): Image save path, if None then display
        title (str): Chart title, default "Training Loss Curve (Batch Level)"

    Returns:
        None: Display or save chart directly, no return value
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, len(loss_history) + 1)
    ax.plot(x, loss_history, label="total_loss", color="#1f77b4", linewidth=2)
    if sub_losses:
        for k, v in sub_losses.items():
            if v is not None and len(v) == len(loss_history):
                ax.plot(x, v, label=k, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Training Batch")
    ax.set_ylabel("Loss")
    ax.legend()
    _finalize(fig, save_path)


# 2) 多loss分解：按epoch绘制多条线或堆叠面积
def plot_multi_loss_breakdown(
    epochs: Sequence[int],
    task_loss: Sequence[float],
    cont_loss: Sequence[float],
    adv_loss: Sequence[float],
    stacked: bool = False,
    save_path: Optional[str] = None,
    title: str = "Multiple Loss Decomposition (by Epoch)"
) -> None:
    """
    Plot multiple loss decomposition by epoch, supporting line chart and stacked area chart

    Args:
        epochs (Sequence[int]): Epoch sequence for x-axis
        task_loss (Sequence[float]): Task loss sequence
        cont_loss (Sequence[float]): Contrast loss sequence
        adv_loss (Sequence[float]): Adversarial loss sequence
        stacked (bool): Whether to use stacked area chart, default False (use line chart)
        save_path (Optional[str]): Image save path, if None then display
        title (str): Chart title, default "Multiple Loss Decomposition (by Epoch)"

    Returns:
        None: Display or save chart directly, no return value
    """
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
    ax.set_ylabel("Loss")
    ax.legend(loc="best")
    _finalize(fig, save_path)


# 3) 训练 vs 验证loss
def plot_train_vs_val_loss(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    save_path: Optional[str] = None,
    title: str = "Training vs Validation Loss (Overfitting Check)"
) -> None:
    """
    Plot training loss vs validation loss for overfitting check

    Args:
        train_losses (Sequence[float]): Training loss sequence
        val_losses (Sequence[float]): Validation loss sequence
        save_path (Optional[str]): Image save path, if None then display
        title (str): Chart title, default "Training vs Validation Loss (Overfitting Check)"

    Returns:
        None: Display or save chart directly, no return value
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    x_train = np.arange(1, len(train_losses) + 1)
    x_val = np.arange(1, len(val_losses) + 1)
    ax.plot(x_train, train_losses, label="train_loss", color="#1f77b4", linewidth=2)
    ax.plot(x_val, val_losses, label="val_loss", color="#d62728", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    _finalize(fig, save_path)


# 3.1) 按Epoch绘制：train_loss、val_loss 与 val_AUROC（双y轴）
def _apply_smooth(arr: Sequence[float], method: Optional[str] = None, alpha: float = 0.2, window: int = 3) -> np.ndarray:
    """
    对数值序列应用平滑处理
    
    Args:
        arr (Sequence[float]): 待平滑处理的数值序列
        method (Optional[str]): 平滑方法，可选 None（不平滑）、"ema"（指数移动平均）、"moving"（滑动平均）
        alpha (float): EMA平滑系数，范围在0-1之间，默认为0.2
        window (int): 滑动平均窗口大小，必须大于等于1，默认为3
    
    Returns:
        np.ndarray: 平滑处理后的数组
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
    title: str = "Training/Validation Loss and Validation AUROC Curve by Epoch",
    smooth: Optional[str] = None,
    smooth_alpha: float = 0.2,
    smooth_window: int = 3,
) -> None:
    """
    Plot training loss, validation loss, and optional validation AUROC curve for each epoch

    Args:
        train_losses (Sequence[float]): Training loss sequence for each epoch
        val_losses (Sequence[float]): Validation loss sequence for each epoch
        val_aurocs (Optional[Sequence[float]]): Validation AUROC value sequence for each epoch, optional
        epochs (Optional[Sequence[int]]): Epoch index sequence, default 1 to N if not provided
        save_path (Optional[str]): Image save path, if None then display
        title (str): Chart title, default "Training/Validation Loss and Validation AUROC Curve by Epoch"
        smooth (Optional[str]): Smoothing method, optional None, "ema", "moving"
        smooth_alpha (float): EMA smoothing coefficient, default 0.2
        smooth_window (int): Moving average window size, default 3

    Returns:
        None: Display or save chart directly, no return value

    Raises:
        ValueError: When training loss and validation loss are empty or have inconsistent lengths
        ValueError: When epochs length is inconsistent with loss sequence length
        ValueError: When val_aurocs length is inconsistent with loss sequence length
    """
    tl = np.asarray(train_losses, dtype=np.float64)
    vl = np.asarray(val_losses, dtype=np.float64)
    if len(tl) == 0 or len(vl) == 0:
        raise ValueError("train_losses and val_losses cannot be empty.")
    if len(tl) != len(vl):
        raise ValueError("train_losses and val_losses must have the same length.")

    n = len(tl)
    if epochs is None:
        x = np.arange(1, n + 1)
    else:
        x = np.asarray(epochs, dtype=np.int64)
        if len(x) != n:
            raise ValueError("epochs length must match loss sequence length.")

    # Smoothing (disabled by default)
    tl_s = _apply_smooth(tl, method=smooth, alpha=smooth_alpha, window=smooth_window)
    vl_s = _apply_smooth(vl, method=smooth, alpha=smooth_alpha, window=smooth_window)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    # Left axis: loss
    l1, = ax1.plot(x, tl_s, label="train_loss", color="#1f77b4", linewidth=2, marker="o", markersize=3)
    l2, = ax1.plot(x, vl_s, label="val_loss", color="#d62728", linewidth=2, marker="s", markersize=3)
    lines = [l1, l2]
    labels = ["training loss", "validation loss"]
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(title)

    # Right axis: AUROC (if provided)
    if val_aurocs is not None:
        va = np.asarray(val_aurocs, dtype=np.float64)
        if len(va) != n:
            raise ValueError("val_aurocs length must match loss sequence length.")
        va_s = _apply_smooth(va, method=smooth, alpha=smooth_alpha, window=smooth_window)
        ax2 = ax1.twinx()
        l3, = ax2.plot(x, va_s, label="val_AUROC", color="#2ca02c", linewidth=2, marker="o", markersize=4)
        ax2.set_ylabel("AUROC")
        ax2.set_ylim(-0.02, 1.02)
        lines.append(l3)
        labels.append("validation AUROC")

    # Merge legend to top-left corner
    ax1.legend(lines, labels, loc="best")
    _finalize(fig, save_path)


def plot_epoch_curves_from_df(
    df: Union[pd.DataFrame, List[Dict]],
    cols: Dict[str, str] = {"epoch": "epoch", "train": "loss_train", "val": "val_loss", "auroc": "val_auroc"},
    save_path: Optional[str] = None,
    title: str = "Training/Validation Loss and Validation AUROC Curve by Epoch",
    smooth: Optional[str] = None,
    smooth_alpha: float = 0.2,
    smooth_window: int = 3,
) -> None:
    """
    Plot training loss, validation loss, and optional validation AUROC curve from DataFrame data for each epoch

    Args:
        df (Union[pd.DataFrame, List[Dict]]): DataFrame or dictionary list containing training process data
        cols (Dict[str, str]): Column name mapping dictionary, specifying actual column names in DataFrame,
                              default {"epoch": "epoch", "train": "loss_train", "val": "val_loss", "auroc": "val_auroc"}
        save_path (Optional[str]): Image save path, if None then display
        title (str): Chart title, default "Training/Validation Loss and Validation AUROC Curve by Epoch"
        smooth (Optional[str]): Smoothing method, optional None, "ema", "moving"
        smooth_alpha (float): EMA smoothing coefficient, default 0.2
        smooth_window (int): Moving average window size, default 3

    Returns:
        None: Display or save chart directly, no return value

    Raises:
        ValueError: When DataFrame is missing required columns
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Read epoch
    epoch_col = cols.get("epoch", "epoch")
    if epoch_col not in df.columns:
        raise ValueError(f"DataFrame missing column: {epoch_col}")
    epochs = df[epoch_col].to_numpy()

    # Read training loss
    train_col = cols.get("train", "loss_train")
    if train_col not in df.columns:
        raise ValueError(f"DataFrame missing training loss column: {train_col}")
    train_losses = df[train_col].to_numpy()

    # Read validation loss (required)
    val_col = cols.get("val", "val_loss")
    if val_col not in df.columns:
        raise ValueError(f"DataFrame missing required column: {val_col}")
    val_losses = df[val_col].to_numpy()

    # Read AUROC (optional)
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
    title: str = "Learning Rate Schedule Curve"
) -> None:
    """
    Plot learning rate schedule curve

    Args:
        lrs (Sequence[float]): Learning rate sequence for each epoch
        save_path (Optional[str]): Image save path, if None then display chart
        title (str): Chart title, default "Learning Rate Schedule Curve"

    Returns:
        None: Display or save chart directly, no return value
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(1, len(lrs) + 1)
    ax.plot(x, lrs, color="#9467bd", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    _finalize(fig, save_path)


# 5) 每Epoch指标柱状图（AUROC/AUPRC/F1）
def plot_epoch_metrics_bar(
    epoch_metrics: Union[pd.DataFrame, List[Dict]],
    metrics: List[str] = ["auroc", "auprc", "f1"],
    save_path: Optional[str] = None,
    title: str = "Epoch Metric Summary (Bar Chart)"
) -> None:
    """
    Plot metric bar chart for each epoch

    Args:
        epoch_metrics (Union[pd.DataFrame, List[Dict]]): Data containing metrics for each epoch, can be DataFrame or dictionary list
        metrics (List[str]): List of metrics to plot, default ["auroc", "auprc", "f1"]
        save_path (Optional[str]): Image save path, if None then display chart
        title (str): Chart title, default "Epoch Metric Summary (Bar Chart)"

    Returns:
        None: Display or save chart directly, no return value
    """
    if not isinstance(epoch_metrics, pd.DataFrame):
        epoch_metrics = pd.DataFrame(epoch_metrics)
    fig, ax = plt.subplots(figsize=(12, 5))
    df = epoch_metrics[["epoch"] + metrics].melt(id_vars="epoch", var_name="metric", value_name="value")
    sns.barplot(data=df, x="epoch", y="value", hue="metric", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric Value")
    ax.legend(loc="best")
    _finalize(fig, save_path)


# 6) ROC curve
def plot_roc_curve(
    y_true: Sequence[int],
    y_score: Sequence[float],
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot ROC curve

    Args:
        y_true (Sequence[int]): True label sequence
        y_score (Sequence[float]): Predicted score sequence
        save_path (Optional[str]): Image save path, if None then display chart
        title (Optional[str]): Chart title, default "ROC Curve" if None

    Returns:
        None: Display or save chart directly, no return value
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC (AUROC={auc:.4f})", color="#1f77b4", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title(title or "ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    _finalize(fig, save_path)


# 7) PR curve
def plot_pr_curve(
    y_true: Sequence[int],
    y_score: Sequence[float],
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> None:
    """
    Plot Precision-Recall curve

    Args:
        y_true (Sequence[int]): True label sequence
        y_score (Sequence[float]): Predicted score sequence
        save_path (Optional[str]): Image save path, if None then display chart
        title (Optional[str]): Chart title, default "Precision-Recall Curve" if None

    Returns:
        None: Display or save chart directly, no return value
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label=f"PR (AUPRC={ap:.4f})", color="#ff7f0e", linewidth=2)
    ax.set_title(title or "Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="best")
    _finalize(fig, save_path)


# 8) 校准curve（预测概率 vs. 真实分数）
def plot_calibration_curve(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
    save_path: Optional[str] = None,
    title: str = "Probability Calibration Curve"
) -> None:
    """
    Plot probability calibration curve

    Args:
        y_true (Sequence[int]): True label sequence
        y_prob (Sequence[float]): Predicted probability sequence
        n_bins (int): Number of bins, default 10
        save_path (Optional[str]): Image save path, if None then display chart
        title (str): Chart title, default "Probability Calibration Curve"

    Returns:
        None: Display or save chart directly, no return value
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, "s-", label="Calibration", color="#2ca02c")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Predicted Probability Binning Mean")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="best")
    _finalize(fig, save_path)


# 9) 阈值扫描图（F1 vs. Threshold）
def plot_threshold_scan(
    thresholds: Sequence[float],
    f1_vals: Sequence[float],
    save_path: Optional[str] = None,
    title: str = "F1 vs. Threshold Scanning"
) -> None:
    """
    Plot F1 value vs threshold curve

    Args:
        thresholds (Sequence[float]): Threshold sequence
        f1_vals (Sequence[float]): F1 value sequence corresponding to thresholds
        save_path (Optional[str]): Image save path, if None then display chart
        title (str): Chart title, default "F1 vs. Threshold Scanning"

    Returns:
        None: Display or save chart directly, no return value
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1_vals, color="#d62728", linewidth=2)
    best_idx = int(np.argmax(f1_vals)) if len(f1_vals) > 0 else None
    if best_idx is not None:
        ax.axvline(thresholds[best_idx], color="#d62728", linestyle="--", alpha=0.6, label=f"best={thresholds[best_idx]:.3f}, F1={f1_vals[best_idx]:.4f}")
    ax.set_title(title)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1")
    ax.legend(loc="best")
    _finalize(fig, save_path)


# 10) 温度缩放效果（可靠性图 + ECE 前后对比）
def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    计算期望校准误差 ECE。
    
    Args:
        y_true (np.ndarray): 真实标签数组
        y_prob (np.ndarray): 预测概率数组
        n_bins (int): 分箱数量，默认为10
    
    Returns:
        float: 计算得到的期望校准误差值
    """
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
    title: str = "Temperature Scaling Effect (Reliability/ECE)"
) -> None:
    """
    Plot temperature scaling effect comparison, showing reliability curve and ECE value before/after calibration

    Args:
        y_true (Sequence[int]): True label sequence
        logits (Sequence[float]): Raw logits before sigmoid processing
        T_opt (Optional[float]): Optimal temperature value, if None then no temperature scaling
        n_bins (int): Number of bins, default 10
        save_path (Optional[str]): Image save path, if None then display chart
        title (str): Chart title, default "Temperature Scaling Effect (Reliability/ECE)"

    Returns:
        None: Display or save chart directly, no return value
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

    # Reliability plot
    fig, ax = plt.subplots(figsize=(6, 6))
    bt, bp = calibration_curve(y_true_np, probs_before, n_bins=n_bins, strategy="uniform")
    at, ap = calibration_curve(y_true_np, probs_after, n_bins=n_bins, strategy="uniform")
    ax.plot(bp, bt, "o-", label=f"Before Calibration (ECE={ece_before:.4f})", color="#7f7f7f")
    ax.plot(ap, at, "s-", label=f"After Calibration (ECE={ece_after:.4f}, T={T_opt})", color="#1f77b4")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Predicted Probability Binning Mean")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="best")
    _finalize(fig, save_path)


# 11) 每折性能比较（箱线或小提琴）
def plot_per_fold_comparison(
    fold_results: List[Dict[str, float]],
    use_violin: bool = False,
    metrics: List[str] = ["auroc", "auprc", "f1"],
    save_path: Optional[str] = None,
    title: str = "5-Fold Performance Comparison"
) -> None:
    """
    Plot performance comparison for each fold cross-validation results, supporting box plot and violin plot

    Args:
        fold_results (List[Dict[str, float]]): Evaluation results list for each fold, each element is a dictionary
        use_violin (bool): Whether to use violin plot, default False (use box plot)
        metrics (List[str]): List of metrics to compare, default ["auroc", "auprc", "f1"]
        save_path (Optional[str]): Image save path, if None then display chart
        title (str): Chart title, default "5-Fold Performance Comparison"

    Returns:
        None: Display or save chart directly, no return value
    """
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
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    _finalize(fig, save_path)


# 12) 混淆矩阵热力图
def plot_confusion_matrix_heatmap(
    cm: Union[Tuple[int, int, int, int], np.ndarray],
    normalize: bool = False,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix Heatmap"
) -> None:
    """
    Plot confusion matrix heatmap

    Args:
        cm (Union[Tuple[int, int, int, int], np.ndarray]): Confusion matrix, can be (tn, fp, fn, tp) tuple or 2x2 matrix
        normalize (bool): Whether to normalize, default False
        save_path (Optional[str]): Image save path, if None then display chart
        title (str): Chart title, default "Confusion Matrix Heatmap"

    Returns:
        None: Display or save chart directly, no return value

    Raises:
        ValueError: When confusion matrix is not 2x2 shape
    """
    if isinstance(cm, tuple) or isinstance(cm, list):
        tn, fp, fn, tp = cm
        mat = np.array([[tn, fp], [fn, tp]], dtype=np.float64)
    else:
        mat = np.asarray(cm, dtype=np.float64)
        if mat.shape != (2, 2):
            raise ValueError("Confusion matrix must be 2x2 or (tn, fp, fn, tp).")

    disp = mat.copy()
    if normalize:
        row_sum = disp.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        disp = disp / row_sum

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(disp, annot=True, fmt=".3f" if normalize else "g", cmap="Blues", cbar=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    _finalize(fig, save_path)


# ==================== 辅助加载器（可选） ====================

def load_epoch_metrics_csv(csv_path: str) -> pd.DataFrame:
    """
    读取 train.py 保存的 metrics/train_epoch_metrics_*.csv 文件，并返回处理后的数据帧
    
    该函数读取训练过程中保存的CSV格式指标文件，确保特定列的数据类型正确性，特别
    是将 epoch、tn、fp、fn、tp 列转换为整数类型以保证后续分析的准确性。
    
    参数:
        csv_path (str): CSV文件的路径，该文件应包含训练过程中的各项评估指标
        
    返回:
        pd.DataFrame: 包含以下列的数据框:
            - epoch: 训练轮次
            - loss_train: 训练loss
            - task_loss: 任务loss
            - cont_loss: 对比loss
            - adv_loss: 对抗loss
            - auroc: AUROC评估指标
            - auprc: AUPRC评估指标
            - precision: 精确率
            - recall: 召回率
            - f1: F1分数
            - tn: 真负例数量
            - fp: 假正例数量
            - fn: 假负例数量
            - tp: 真正例数量
    """
    df = pd.read_csv(csv_path)
    # 保证类型正确
    for col in ["epoch", "tn", "fp", "fn", "tp"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def derive_threshold_scan_arrays(txt_path: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    从 threshold_scan_*.txt 文件中解析阈值扫描结果
    
    该函数读取阈值扫描文件，提取最佳阈值、最佳F1分数以及校准后的最佳阈值和F1分数。
    主要用于二分类模型的阈值优化分析，支持原始和温度校准后的阈值比较。
    
    参数:
        txt_path (str): 阈值扫描结果文件路径，该文件应包含阈值扫描的统计信息
        
    返回:
        Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]: 
        四元组包含以下元素:
            - best_t: 最佳阈值
            - best_f1: 最佳F1分数
            - best_t_cal: 校准后的最佳阈值
            - best_f1_cal: 校准后的最佳F1分数
        如果文件不存在或解析失败，相应位置的值将为None
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