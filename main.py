import os, sys  # 系统交互（环境变量、路径等）
# 让以 'python model/main.py' 运行时也能按包导入
try:
    ROOT = os.path.dirname(os.path.dirname(__file__))
    if ROOT and ROOT not in sys.path:
        sys.path.insert(0, ROOT)
except Exception:
    pass
import torch  # PyTorch 深度学习框架
import numpy as np  # 科学计算与数组操作
from parms_setting import settings  # 参数与超参数设置
from utils import set_global_seed  # 随机性统一设置
from data_preprocess import load_data, get_fold_data  # 数据加载与折叠划分
from instantiation import Create_model  # 模型实例化
from train import train_model  # 训练流程

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

# 初始化集中日志（文件+控制台），日志开头记录完整命令
# 先解析全部参数（含 run_name、shutdown）
args = settings()
# 验证早停设置：若未在参数中提供则注入默认值，使阶段B/C训练受益
if not hasattr(args, "early_stop_patience"):
    args.early_stop_patience = 3
if not hasattr(args, "early_stop_min_delta"):
    args.early_stop_min_delta = 0.0
# 采用验证集 AUPRC 作为早停指标（train_model 内部应读取 args 使用；未使用则不影响）
if not hasattr(args, "early_stop_metric"):
    args.early_stop_metric = "auprc"



# 已移除自动并行环境初始化（按反馈不要性能增强）

# 初始化集中日志（文件+控制台），带 run_name
logger = init_logging(run_name=args.run_name)
# 重定向所有 print 到日志，同时保留控制台输出
redirect_print(True)
# 创建当前运行结果目录（data_时间戳）并记录
make_result_run_dir("data")
logger.info("Initialized logging and result directory.")


# 将后续 print 重定向到 logger.info，避免控制台重复输出
def _print_to_logger(*args, **kwargs):
    try:
        msg = " ".join(str(x) for x in args)
    except Exception:
        msg = " ".join(map(str, args))
    logger.info(msg)
print = _print_to_logger

# 设置程序使用的GPU设备
# "CUDA_VISIBLE_DEVICES"是一个环境变量，用于指定哪些GPU可以被CUDA应用程序看到
# "0"表示只使用系统中编号为0的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# parameters setting  # 注释：参数设置（已提前解析，此处无需重复）
# args 已在日志初始化前由 settings() 获取

# 固定使用 CUDA，不再进行可用性检查
args.cuda = True
logger.info("Using CUDA: True")
logger.info(f"CUDA device count: {torch.cuda.device_count()}")
logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
# 统一随机性：一次性在入口设置（含 Python/NumPy/PyTorch/CUDA/cuDNN）
set_global_seed(int(getattr(args, "seed", 0)))

# 打印增强配置汇总
try:
    _aug = getattr(args, "augment", "random_permute_features")
    _mode = getattr(args, "augment_mode", "static") if hasattr(args, "augment_mode") else "static"
    _noise = getattr(args, "noise_std", 0.01)
    _mask = getattr(args, "mask_rate", 0.1)
    _aseed = getattr(args, "augment_seed", None)
    logger.info("=== Augmentation Config ===")
    try:
        _aug_str = ",".join(_aug) if isinstance(_aug, (list, tuple)) else str(_aug)
    except Exception:
        _aug_str = str(_aug)
    # 额外的人类可读提示：单/多增强
    try:
        if isinstance(_aug, (list, tuple)):
            print(f"[AUG CONFIG] Multiple augmentations detected: {', '.join(map(str, _aug))}")
        else:
            print(f"[AUG CONFIG] Single augmentation: {_aug}")
    except Exception:
        pass
    logger.info(f"augment={_aug_str} | mode={_mode} | noise_std={_noise} | mask_rate={_mask} | augment_seed={_aseed} (None means seed+fold for static)")
    logger.info("===========================")
except Exception as _e:
    logger.info(f"[AUGMENT] config print skipped due to: {_e}")


# load data  # 注释：加载数据
# 调用load_data函数，传入参数对象args
# 该函数会返回处理好的图数据对象（原始图和对抗图）以及所有折的训练和测试数据加载器
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
    aurocs = [result['auroc'] for result in all_fold_results]
    auprcs = [result['auprc'] for result in all_fold_results]
    f1s = [result['f1'] for result in all_fold_results]
    losses = [result['loss'] for result in all_fold_results]
    
    logger.info(f"AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    logger.info(f"AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
    logger.info(f"F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    logger.info(f"Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    # 保存最终 5-fold 汇总指标到 result_summary_{run_id}.txt（与日志后缀一致）
    _paths = get_run_paths()
    _run_id = _paths.get("run_id") or ""
    _summary_lines = [
        "5-Fold Cross Validation Summary",
        f"AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}",
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
    precisions = [result.get('precision', 0.0) for result in all_fold_results]
    recalls = [result.get('recall', 0.0) for result in all_fold_results]
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
try:
    from log_output_manager import get_run_paths, make_result_run_dir
    _paths = get_run_paths()
    _run_dir = _paths.get("run_result_dir") or str(make_result_run_dir("data"))
    _run_id = _paths.get("run_id") or os.path.basename(_run_dir)

    _metrics_dir_em = os.path.join(_run_dir, "metrics")
    # 1) 每折：训练epoch指标（loss/分项/auroc/auprc/f1）
    for fold in range(1, 6):
        try:
            csv_name = f"train_epoch_metrics_fold_{fold}_{_run_id}.csv"
            csv_path = os.path.join(_metrics_dir_em, csv_name)
            if os.path.exists(csv_path):
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
        except Exception as _e:
            logger.warning(f"[VIS] fold {fold} epoch metrics plot skipped: {_e}")

    # 2) 测试阶段曲线：从 OUTPUT/result/metrics 读取
    _metrics_dir_out = os.path.join(_run_dir, "metrics")
    for fold in range(1, 6):
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
            logger.info(f"[VIS] saved: {os.path.abspath(p)}")
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