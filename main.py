import os
import sys

# 让以 'python model/main.py' 运行时也能按包导入
try:
    ROOT = os.path.dirname(os.path.dirname(__file__))
    if ROOT and ROOT not in sys.path:
        sys.path.insert(0, ROOT)
except Exception:
    pass

import torch
import numpy as np

# 参数与超参数设置
from parms_setting import settings
# 随机性统一设置
from utils import set_global_seed
# 数据加载与折叠划分
from data_preprocess import load_data, get_fold_data
# 模型实例化
from instantiation import Create_model
# 训练流程
from train import train_model
# 日志管理
from log_output_manager import *

# 可视化
from visualization import (
    load_epoch_metrics_csv,
    plot_multi_loss_breakdown,
    plot_epoch_metrics_bar,
    plot_train_vs_val_loss,
    plot_epoch_curves_from_df,
    plot_roc_curve,
    plot_pr_curve,
    plot_calibration_curve,
    plot_temperature_scaling_effect,
    plot_threshold_scan,
    plot_per_fold_comparison,
    plot_confusion_matrix_heatmap
)


# 参数改由 EM/parms_setting.py 统一解析（包含 --run_name 与 --shutdown）

# 性能优化相关函数已迁移至 autodl.py

args = settings()
# 如果args中没有early_stop_patience属性，则设置默认值为3
if not hasattr(args, "early_stop_patience"):
    args.early_stop_patience = 3
# 如果args中没有early_stop_min_delta属性，则设置默认值为0.0
if not hasattr(args, "early_stop_min_delta"):
    args.early_stop_min_delta = 0.0
# 如果args中没有early_stop_metric属性，则设置默认值为"auprc"
if not hasattr(args, "early_stop_metric"):
    args.early_stop_metric = "auprc"

# 获取CUDA可见设备设置，默认为"0"
cuda_visible_devices = getattr(args, "cuda_visible_devices", "0")
# 设置环境变量CUDA_VISIBLE_DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


# 初始化日志记录器
logger = init_logging(run_name=args.run_name)
# 重定向print函数输出到日志
redirect_print(True)
# 创建结果运行目录
make_result_run_dir("data")
logger.info("Initialized logging and result directory.")

def _print_to_logger(*args, **kwargs):
    """
    将print函数的输出重定向到日志记录器
    
    Args:
        *args: 任意数量的位置参数，要打印的消息
        **kwargs: 任意数量的关键字参数
    """
    try:
        msg = " ".join(str(x) for x in args)
    except Exception:
        msg = " ".join(map(str, args))
    logger.info(msg)
# 重写内置print函数，使其输出到日志记录器
print = _print_to_logger


# 检查CUDA是否可用，并设置args.cuda属性
args.cuda = torch.cuda.is_available()
if args.cuda:
    # CUDA可用时记录相关信息
    logger.info("Using CUDA: True")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
    try:
        logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
    except Exception:
        logger.info("Could not get CUDA device name")
else:
    # CUDA不可用时记录相关信息
    logger.info("Using CUDA: False - CUDA is not available")
# 设置全局随机种子
set_global_seed(int(getattr(args, "seed", 0)))

# 打印增强配置汇总
try:
    aug = getattr(args, "augment", "random_permute_features")
    mode = getattr(args, "augment_mode", "static") if hasattr(args, "augment_mode") else "static"
    noise = getattr(args, "noise_std", 0.01)
    mask = getattr(args, "mask_rate", 0.1)
    aseed = getattr(args, "augment_seed", None)
    logger.info("=== Augmentation Config ===")
    
    try:
        aug_str = ",".join(aug) if isinstance(aug, (list, tuple)) else str(aug)
    except Exception:
        aug_str = str(aug)
        
    try:
        if isinstance(aug, (list, tuple)):
            print(f"[AUG CONFIG] Multiple augmentations detected: {', '.join(map(str, aug))}")
        else:
            print(f"[AUG CONFIG] Single augmentation: {aug}")
    except Exception:
        pass
        
    logger.info(f"augment={aug_str} | mode={mode} | noise_std={noise} | mask_rate={mask} | augment_seed={aseed} (None means seed+fold for static)")
    logger.info("===========================")
except Exception as e:
    logger.info(f"[AUGMENT] config print skipped due to: {e}")


data_o_folds, data_a_folds, train_loaders, test_loaders = load_data(args)

# 存储每一折的结果
all_fold_results = []
logger.info("Starting 5-fold cross validation...")

for fold in range(5):
    # 按折使用对应的图数据与加载器
    data_o = data_o_folds[fold]
    data_a = data_a_folds[fold]
    logger.info(f"=== Fold {fold + 1}/5 ===")
    
    # 为每一折创建新的模型和优化器
    model, optimizer = Create_model(args)
    
    # 获取当前折的数据加载器
    train_loader = train_loaders[fold]
    test_loader = test_loaders[fold]
    
    # 训练和测试当前折的模型
    fold_results = train_model(model, optimizer, data_o, data_a, train_loader, test_loader, args, fold_idx=fold+1)
    all_fold_results.append(fold_results)
    
    logger.info(f"Fold {fold + 1} completed.")

# 计算所有折的平均结果
logger.info("=== 5-Fold Cross Validation Results ===")
if all_fold_results:
    # 提取各折的评估指标
    aurocs = [result['auroc'] for result in all_fold_results]
    auprcs = [result['auprc'] for result in all_fold_results]
    f1s = [result['f1'] for result in all_fold_results]
    losses = [result['loss'] for result in all_fold_results]
    
    # 输出平均指标及标准差
    logger.info(f"AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    logger.info(f"AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
    logger.info(f"F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    logger.info(f"Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    # 保存最终 5-fold 汇总指标到 result_summary_{run_id}.txt（与日志后缀一致）
    _paths = get_run_paths()
    _run_id = _paths.get("run_id") or ""
    _summary_lines = [
        "5-Fold Cross Validation Summary",
        f"AUROC: {np.mean(aurocs):.4f} ± {np.std(auprcs):.4f}",
        f"AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}",
        f"F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
        f"Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}"
    ]
    _fname = f"result_summary_{_run_id}.txt" if _run_id else "result_summary.txt"
    save_result_text("\n".join(_summary_lines), filename=_fname)
    logger.info("Detailed Results:")
    _per_fold_lines = []
    for i, result in enumerate(all_fold_results):
        logger.info(f"Fold {i+1}: AUROC={result['auroc']:.4f}, AUPRC={result['auprc']:.4f}, F1={result['f1']:.4f}")
        _per_fold_lines.append(f"Fold {i+1}: AUROC={result['auroc']:.4f}, AUPRC={result['auprc']:.4f}, F1={result['f1']:.4f}")
    _pfname = f"per_fold_{_run_id}.txt" if _run_id else "per_fold.txt"
    if len(_per_fold_lines) > 0:
        save_result_text("\n".join(_per_fold_lines), filename=_pfname)
else:
    logger.info("No results collected.")
# 追加精度/召回与混淆矩阵的最终汇总保存到 EM/result/metrics，并打印
try:
    # 计算精确率和召回率的平均值及标准差
    precisions = [result.get('precision', 0.0) for result in all_fold_results]
    recalls = [result.get('recall', 0.0) for result in all_fold_results]
    # 累加所有折的混淆矩阵
    cm_sum = np.array([0,0,0,0], dtype=np.int64)
    for result in all_fold_results:
        tn, fp, fn, tp = result.get('cm', (0,0,0,0))
        cm_sum += np.array([tn, fp, fn, tp], dtype=np.int64)
    _paths = get_run_paths()
    _run_id = _paths.get("run_id") or ""
    _extra_lines = [
        "5-Fold Extra Metrics",
        f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
        f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
        f"Confusion Matrix (sum): tn={cm_sum[0]}, fp={cm_sum[1]}, fn={cm_sum[2]}, tp={cm_sum[3]}"
    ]
    save_result_text("\n".join(_extra_lines), filename=f"result_extra_{_run_id}.txt" if _run_id else "result_extra.txt", subdir="metrics")
    logger.info(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    logger.info(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    logger.info(f"Confusion Matrix (sum): tn={cm_sum[0]}, fp={cm_sum[1]}, fn={cm_sum[2]}, tp={cm_sum[3]}")
except Exception as _e:
    logger.warning(f"Failed to save extra metrics: {_e}")

# ===== 自动生成可视化图像输出到 OUTPUT/result 下 =====
def plot_fold_epoch_metrics(fold, _run_id, _metrics_dir_em):
    """
    绘制指定折数的训练epoch指标图表
    
    该函数会查找对应折数的训练指标CSV文件，如果找到则生成多种可视化图表，
    包括损失分解图、各项指标柱状图以及训练曲线图等。
    
    Args:
        fold (int): 当前处理的折数（1-5）
        _run_id (str): 运行标识符，用于定位特定运行的文件
        _metrics_dir_em (str): 指标文件所在目录路径
    """
    try:
        csv_name = f"train_epoch_metrics_fold_{fold}_{_run_id}.csv"
        candidates = [
            os.path.join(_metrics_dir_em, csv_name),
            os.path.join(_metrics_dir_em, csv_name + ".txt"),
            os.path.join(_metrics_dir_em, f"train_epoch_metrics_fold_{fold}.csv"),
            os.path.join(_metrics_dir_em, f"train_epoch_metrics_fold_{fold}.csv.txt"),
        ]
        csv_path = next((p for p in candidates if os.path.exists(p)), None)
        if csv_path:
            df = load_epoch_metrics_csv(csv_path)
            # 多损失分解
            plot_multi_loss_breakdown(df["epoch"].tolist(), df["task_loss"].tolist(), df["cont_loss"].tolist(), df["adv_loss"].tolist(),
                                      stacked=False, save_path=f"loss_breakdown_fold_{fold}.png")
            # 每epoch指标柱状
            plot_epoch_metrics_bar(df, metrics=["auroc", "auprc", "f1"],
                                   save_path=f"epoch_metrics_bar_fold_{fold}.png")
            # 按Epoch三曲线：train_loss/val_loss/val_AUROC（双y轴）
            plot_epoch_curves_from_df(
                df,
                save_path=f"epoch_curves_fold_{fold}.png",
                title="按Epoch的训练/验证损失与验证AUROC曲线"
            )
        else:
            logger.warning(f"[VIS] epoch CSV not found for fold={fold} in {_metrics_dir_em}")
    except Exception as _e:
        logger.warning(f"[VIS] fold {fold} epoch metrics plot skipped: {_e}")

def plot_fold_test_curves(fold, _run_id, _metrics_dir_out):
    """
    绘制指定折数的测试阶段曲线图表
    
    该函数会生成ROC曲线、PR曲线、校准曲线等多种评估图表，
    并检查是否存在温度缩放参数来绘制温度缩放效果图。
    
    Args:
        fold (int): 当前处理的折数（1-5）
        _run_id (str): 运行标识符，用于定位特定运行的文件
        _metrics_dir_out (str): 输出指标文件所在目录路径
    """
    fold_tag = f"fold_{fold}"
    try:
        arr_csv = os.path.join(_metrics_dir_out, f"y_true_pred_{fold_tag}_{_run_id}.csv")
        if os.path.exists(arr_csv):
            import pandas as _pd
            arr_df = _pd.read_csv(arr_csv)
            y_true = arr_df["y_true"].astype(int).tolist()
            y_prob = arr_df["y_prob"].astype(float).tolist()
            logits = arr_df["logit"].astype(float).tolist()
            # ROC / PR / 校准
            plot_roc_curve(y_true, y_prob, save_path=f"roc_fold_{fold}.png")
            plot_pr_curve(y_true, y_prob, save_path=f"pr_fold_{fold}.png")
            plot_calibration_curve(y_true, y_prob, save_path=f"calibration_fold_{fold}.png")
            # 温度缩放效果（若有最佳T）
            import json as _json
            T_json = os.path.join(_metrics_dir_out, f"temperature_{fold_tag}_{_run_id}.json")
            T_opt = None
            if os.path.exists(T_json):
                with open(T_json, "r", encoding="utf-8") as f:
                    T_opt = float(_json.load(f).get("best_T"))
            plot_temperature_scaling_effect(y_true, logits, T_opt, save_path=f"temperature_effect_fold_{fold}.png")
        # 阈值扫描
        th_csv = os.path.join(_metrics_dir_out, f"threshold_scan_{fold_tag}_{_run_id}.csv")
        if os.path.exists(th_csv):
            import pandas as _pd
            th_df = _pd.read_csv(th_csv)
            plot_threshold_scan(th_df["threshold"].tolist(), th_df["f1"].tolist(),
                                save_path=f"threshold_scan_fold_{fold}.png")
        th_cal_csv = os.path.join(_metrics_dir_out, f"threshold_scan_calibrated_{fold_tag}_{_run_id}.csv")
        if os.path.exists(th_cal_csv):
            import pandas as _pd
            th_df2 = _pd.read_csv(th_cal_csv)
            plot_threshold_scan(th_df2["threshold"].tolist(), th_df2["f1_cal"].tolist(),
                                save_path=f"threshold_scan_calibrated_fold_{fold}.png",
                                title="F1 vs. 阈值扫描（温度校准后）")
    except Exception as _e:
        logger.warning(f"[VIS] fold {fold} test curves plot skipped: {_e}")

try:
    from log_output_manager import get_run_paths, make_result_run_dir
    _paths = get_run_paths()
    _run_dir = _paths.get("run_result_dir") or str(make_result_run_dir("data"))
    _run_id = _paths.get("run_id") or os.path.basename(_run_dir)

    _metrics_dir_em = os.path.join(_run_dir, "metrics")
    # 1) 每折：训练epoch指标（loss/分项/auroc/auprc/f1）
    for fold in range(1, 6):
        plot_fold_epoch_metrics(fold, _run_id, _metrics_dir_em)

    # 2) 测试阶段曲线：从 OUTPUT/result/metrics 读取
    _metrics_dir_out = os.path.join(_run_dir, "metrics")
    for fold in range(1, 6):
        plot_fold_test_curves(fold, _run_id, _metrics_dir_out)

    # 3) 每折性能比较（箱线）
    try:
        plot_per_fold_comparison(all_fold_results, use_violin=False,
                                 save_path="per_fold_box.png")
    except Exception as _e:
        logger.warning(f"[VIS] per-fold comparison skipped: {_e}")

    # 4) 混淆矩阵热力图（合计）
    try:
        import numpy as _np
        cm_sum = _np.array([0,0,0,0], dtype=_np.int64)
        for result in all_fold_results:
            tn, fp, fn, tp = result.get('cm', (0,0,0,0))
            cm_sum += _np.array([tn, fp, fn, tp], dtype=_np.int64)
        plot_confusion_matrix_heatmap(tuple(cm_sum.tolist()), normalize=False,
                                      save_path="confusion_matrix_sum.png",
                                      title="混淆矩阵（5折合计）")
    except Exception as _e:
        logger.warning(f"[VIS] confusion matrix plot skipped: {_e}")

    try:
        _fig_dir = os.path.join(_run_dir, "figure")
        files = sorted([os.path.join(_fig_dir, f) for f in os.listdir(_fig_dir) if f.lower().endswith(".png")])
        for p in files:
            logger.debug(f"[VIS] saved: {os.path.abspath(p)}")
        logger.info(f"[VIS] All plots saved to {_fig_dir}")
    except Exception as _e_list:
        logger.warning(f"[VIS] list saved files failed: {_e_list}")
    # 生成 metrics 清单（CSV/JSON）
    try:
        _metrics_dir_out = os.path.join(_run_dir, "metrics")
        os.makedirs(_metrics_dir_out, exist_ok=True)
        manifest_path = os.path.join(_metrics_dir_out, "files_manifest.txt")
        items = []
        for fname in sorted(os.listdir(_metrics_dir_out)):
            if fname.lower().endswith(".csv") or fname.lower().endswith(".json"):
                items.append(os.path.abspath(os.path.join(_metrics_dir_out, fname)))
        with open(manifest_path, "w", encoding="utf-8") as f:
            for p in items:
                f.write(p + "\n")
        logger.info(f"[VIS] metrics manifest saved: {os.path.abspath(manifest_path)} ({len(items)} items)")
    except Exception as _e_manifest:
        logger.warning(f"[VIS] metrics manifest failed: {_e_manifest}")
except Exception as _e:
    logger.warning(f"[VIS] auto plotting failed: {_e}")

logger.info("All folds completed!")
# 记录运行结束并（在 Linux 且命令指定时）执行关机
finalize_run()
perform_shutdown_if_linux(args.shutdown)