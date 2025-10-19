import os
import platform
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# ========= 可选依赖（仅用于 CPU 亲和） =========
try:
    import psutil  # 可选：用于 CPU 亲和
except Exception:
    psutil = None


# ========= Linux 专用：NUMA 和 CPU 亲和 =========
def _detect_linux_numa_node0_cpus() -> Optional[List[int]]:
    """在 Linux 上探测 node0 的 CPU 列表（若不可用则返回 None）"""
    try:
        nodes_path = "/sys/devices/system/node"
        if not os.path.isdir(nodes_path):
            return None
        node0 = os.path.join(nodes_path, "node0")
        if not os.path.isdir(node0):
            return None
        cpu_list = []
        for name in os.listdir(node0):
            if name.startswith("cpu") and name[3:].isdigit():
                cpu_list.append(int(name[3:]))
        return sorted(cpu_list) if cpu_list else None
    except Exception:
        return None


def _set_cpu_affinity_linux(cpus: List[int]) -> bool:
    """尝试设置当前进程的 CPU 亲和（Linux）。成功返回 True，否则 False。"""
    try:
        if psutil is not None:
            p = psutil.Process(os.getpid())
            p.cpu_affinity(cpus)
            return True
    except Exception:
        pass
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(cpus))
            return True
    except Exception:
        pass
    return False


# ========= 并行与 DataLoader 工具 =========
def _derive_workers(threads: int, req_workers: int) -> int:
    """根据线程上限与请求值派生 dataloader 的 workers 数"""
    if req_workers == -1:
        return min(8, max(1, threads))
    return max(0, min(32, req_workers))


def decide_dataloader_workers(args: object) -> int:
    """
    统一决策 DataLoader 的 num_workers：
    - Windows 固定为 0（避免多进程问题）
    - 其他平台：若未指定（-1），按线程上限自动推导，范围 [1,8]
    - 若显式指定（>=0），在 [0,32] 截断
    """
    os_name = platform.system().lower()
    try:
        threads = int(getattr(args, "threads", 32) or 32)
    except Exception:
        threads = 32
    req_workers_attr = getattr(args, "num_workers", -1)
    try:
        req_workers = int(req_workers_attr if req_workers_attr is not None else -1)
    except Exception:
        req_workers = -1

    if os_name.startswith("win"):
        return 0
    return _derive_workers(threads, req_workers)


def setup_parallelism(threads: int) -> None:
    """
    统一设置数值后端线程数（不修改 torch.set_num_threads，避免影响 GPU）。
    会设置以下环境变量：
    - OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS, NUMEXPR_NUM_THREADS, VECLIB_MAXIMUM_THREADS, BLIS_NUM_THREADS
    - 若在 Linux 且开启 EM_USE_NUMA 或 EM_CPU_AFFINITY，则尝试设置 CPU 亲和绑定到 node0 或限制到前 t 个核
    """
    t = int(max(1, min(32, threads)))
    for k in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[k] = str(t)
    try:
        if platform.system().lower() == "linux":
            use_aff = os.environ.get("EM_USE_NUMA") == "1" or os.environ.get("EM_CPU_AFFINITY") == "1"
            if use_aff:
                cpus = _detect_linux_numa_node0_cpus()
                if not cpus:
                    total = os.cpu_count() or 32
                    cpus = list(range(min(t, total)))
                _ = _set_cpu_affinity_linux(cpus)
    except Exception:
        pass


# ========= 环境初始化入口 =========
def init_autodl_env(args: object) -> None:
    """
    ✅ 修复7：完善函数尾部，确保 setup_parallelism 调用
    一次性初始化项目的性能优化环境：
    - Linux 默认开启 NUMA/亲和（EM_USE_NUMA=1，若未显式指定）
    - 设置并行后端线程（最多 32）
    - 注入 EM_THREADS/EM_WORKERS/EM_CHUNK_SIZE 环境变量，供下游使用
    """
    # Linux 默认启用 NUMA/亲和开关（仅影响CPU亲和，不影响GPU）
    try:
        if platform.system().lower() == "linux":
            if os.environ.get("EM_USE_NUMA") is None and os.environ.get("EM_CPU_AFFINITY") is None:
                os.environ["EM_USE_NUMA"] = "1"
    except Exception:
        pass

    # 统一并行线程设置
    try:
        _threads = int(getattr(args, "threads", 32))
    except Exception:
        _threads = 32
    setup_parallelism(_threads)

    # 将关键并行参数同步到环境变量（workers 默认跟随 threads，cap 32，chunk 默认 20000）
    try:
        req_workers = getattr(args, "num_workers", -1)
        _workers = _derive_workers(_threads, int(req_workers if req_workers is not None else -1))
        _chunk = int(getattr(args, "chunk_size", 0))
        if _chunk in (0, None):
            _chunk = 20000
        os.environ["EM_THREADS"] = str(min(32, max(1, _threads)))
        os.environ["EM_WORKERS"] = str(min(32, max(0, _workers)))
        os.environ["EM_CHUNK_SIZE"] = str(max(1, _chunk))
    except Exception:
        # 兜底
        os.environ.setdefault("EM_THREADS", "32")
        os.environ.setdefault("EM_WORKERS", "8")
        os.environ.setdefault("EM_CHUNK_SIZE", "20000")


# ================== 以下为 A→B→C 自动化驱动实现 ==================
# 约定：复用 main.py 的集中日志与结果目录；每个 trial 通过 run_name 关联输出
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "OUTPUT"
RESULT_DIR = OUTPUT_DIR / "result"
LOG_DIR = OUTPUT_DIR / "log"
EXP_DIR = BASE_DIR.parent / "experiments"


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run_main(run_args: List[str], env: Optional[Dict[str, str]] = None) -> Tuple[int, float, str]:
    """
    以子进程方式运行 main.py，一直将输出直接打到当前控制台。
    返回 (returncode, elapsed_seconds, command_line)
    """
    cmd = [sys.executable, str(BASE_DIR / "main.py")] + run_args
    print(f"[RUN] {' '.join(cmd)}")
    t0 = time.time()
    # 直接继承父进程的 stdout/stderr，实现全量控制台打印
    rc = subprocess.call(cmd, env=env or os.environ.copy())
    elapsed = time.time() - t0
    return rc, elapsed, " ".join(cmd)


def _find_run_result_dir(run_name: str) -> Optional[Path]:
    """
    main.py 使用 log_output_manager.make_result_run_dir:
    目录命名为 OUTPUT/result/{run_name}_data_{run_id}
    这里依据 run_name 前缀寻找最新的匹配目录
    """
    if not RESULT_DIR.exists():
        return None
    candidates = [p for p in RESULT_DIR.iterdir() if p.is_dir() and p.name.startswith(f"{run_name}_data_")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _parse_result_summary(run_dir: Path) -> Dict[str, Any]:
    """
    从 run_dir 下解析 result_summary_*.txt 的 AUROC/AUPRC/F1 mean/std
    """
    if not run_dir or not run_dir.exists():
        return {}
    files = list(run_dir.glob("result_summary_*.txt")) + list(run_dir.glob("result_summary.txt"))
    if not files:
        return {}
    # 取最新
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    f = files[0]
    metrics = {}
    try:
        txt = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in txt:
            line = line.strip()
            for key in ("AUROC", "AUPRC", "F1-Score", "Loss"):
                if line.startswith(key + ":"):
                    # 形如 AUROC: 0.8123 ± 0.0123
                    try:
                        val = line.split(":")[1].strip()
                        mean_str, std_str = val.split("±")
                        metrics[key.lower().replace("-score", "").replace("-", "_") + "_mean"] = float(mean_str)
                        metrics[key.lower().replace("-score", "").replace("-", "_") + "_std"] = float(std_str)
                    except Exception:
                        pass
    except Exception:
        pass
    return metrics


def _write_json(path: Path, obj: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_line(path: Path, line: str) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def _trial_row(task: str, run_name: str, cfg: Dict[str, Any], m: Dict[str, Any], elapsed: float, run_dir: Optional[Path], cmd: str) -> Dict[str, Any]:
    return {
        "task": task,
        "run_name": run_name,
        "config": cfg,
        "auprc_mean": m.get("auprc_mean"),
        "auprc_std": m.get("auprc_std"),
        "auroc_mean": m.get("auroc_mean"),
        "auroc_std": m.get("auroc_std"),
        "f1_mean": m.get("f1_mean"),
        "f1_std": m.get("f1_std"),
        "loss_mean": m.get("loss_mean"),
        "loss_std": m.get("loss_std"),
        "elapsed_sec": round(elapsed, 2),
        "result_dir": str(run_dir) if run_dir else None,
        "command": cmd
    }


# ========== 搜索空间与采样 ==========
COARSE_SPACE = {
    "embed_dim": [32, 64, 128, 256],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "weight_decay": [0.0, 5e-5, 5e-4],
    "dropout": [0.0, 0.1, 0.2],
    "batch": [16, 32, 64],
    "moco_queue": [1024, 4096, 8192],
    "moco_momentum": [0.95, 0.99, 0.999],
    "moco_t": [0.07, 0.1, 0.2],
    "proj_dim": [32, 64, 128, None],
    # 固定为三种增强（列表形式，供下游直接拼接为逗号分隔）
    "augment": [["random_permute_features", "attribute_mask", "noise_then_mask"]],
    # 将增强模式也纳入搜索
    "augment_mode": ["static", "online"],
    "noise_std": [0.005, 0.01, 0.02],
    "mask_rate": [0.05, 0.1, 0.2],
    "loss_ratio1": [0.5, 1.0, 2.0],
    "loss_ratio2": [0.1, 0.5, 1.0],
    "loss_ratio3": [0.0, 0.1, 0.5],
}

# 三任务（LDA/MDA/LMI）专用搜索空间：仅覆盖约定的 CLI 超参
TASK_SEARCH_SPACE = {
    "LDA": {
        "embed_dim": [64, 96, 128],
        "learning_rate": [1e-3, 5e-4, 3e-4, 1e-4],
        "weight_decay": [1e-5, 5e-5, 1e-4, 5e-4],
        "similarity_threshold": [0.4, 0.5, 0.6],
        # alpha/beta/gamma -> loss_ratio1/2/3 候选组
        "abg": [
            (0.5, 0.5, 0.5),
            (0.7, 0.5, 0.3),
            (0.3, 0.5, 0.7),
            (0.6, 0.2, 0.2),
            (0.2, 0.6, 0.2),
            (0.2, 0.2, 0.6),
            (0.4, 0.4, 0.2),
        ],
    },
    "MDA": {
        "embed_dim": [64, 128],
        "learning_rate": [1e-3, 5e-4, 3e-4, 1e-4],
        "weight_decay": [1e-5, 5e-5, 1e-4, 3e-4, 5e-4],
        "similarity_threshold": [0.4, 0.5, 0.6],
        "abg": [
            (0.5, 0.5, 0.5),
            (0.6, 0.5, 0.4),
            (0.4, 0.5, 0.6),
            (0.55, 0.45, 0.5),
            (0.45, 0.55, 0.5),
        ],
    },
    "LMI": {
        "embed_dim": [64, 96],
        "learning_rate": [1e-3, 5e-4, 3e-4, 1e-4],
        "weight_decay": [1e-5, 5e-5, 1e-4, 5e-4],
        "similarity_threshold": [0.4, 0.5],
        "abg": [
            (0.5, 0.5, 0.5),
            (0.6, 0.5, 0.4),
            (0.4, 0.5, 0.6),
        ],
    },
}

# epochs 候选作为搜索空间的一部分（用于阶段B随机搜索）
EPOCHS_CAND_DEFAULT = [3, 50, 100]


def _rand_choice(seq: List[Any]) -> Any:
    import random
    return seq[random.randrange(0, len(seq))]


def sample_coarse_configs(n: int, task: str, base_seed: int = 0) -> List[Dict[str, Any]]:
    """
    按任务采样“仅定义的搜索空间”超参，其余保持原有随机项不变。
    - embed_dim/lr/weight_decay/similarity_threshold/alpha-beta-gamma 走任务专用空间
    - 其余（如 dropout/batch/augment 等）保持 COARSE_SPACE 原有采样逻辑（不改策略）
    """
    import random
    random.seed(base_seed)
    tspace = TASK_SEARCH_SPACE.get(task, TASK_SEARCH_SPACE["LDA"])
    cfgs: List[Dict[str, Any]] = []
    for _ in range(n):
        cfg: Dict[str, Any] = {}
        # 任务专用采样
        ed = _rand_choice(tspace["embed_dim"])
        cfg["dimensions"] = int(ed)
        cfg["lr"] = float(_rand_choice(tspace["learning_rate"]))
        cfg["weight_decay"] = float(_rand_choice(tspace["weight_decay"]))
        cfg["similarity_threshold"] = float(_rand_choice(tspace["similarity_threshold"]))
        a, b, g = _rand_choice(tspace["abg"])
        cfg["loss_ratio1"] = float(a)
        cfg["loss_ratio2"] = float(b)
        cfg["loss_ratio3"] = float(g)
        # 结构相关与其余随机项（沿用原逻辑）
        cfg["hidden1"] = _rand_choice([ed // 2, ed, ed * 2])
        cfg["hidden2"] = _rand_choice([max(1, ed // 4), max(1, ed // 2), ed])
        cfg["dropout"] = _rand_choice(COARSE_SPACE["dropout"])
        cfg["batch"] = _rand_choice(COARSE_SPACE["batch"])
        cfg["moco_queue"] = _rand_choice(COARSE_SPACE["moco_queue"])
        cfg["moco_momentum"] = _rand_choice(COARSE_SPACE["moco_momentum"])
        cfg["moco_t"] = _rand_choice(COARSE_SPACE["moco_t"])
        pd = _rand_choice(COARSE_SPACE["proj_dim"])
        cfg["proj_dim"] = None if pd is None else int(pd)
        # 固定三增强 + 模式
        cfg["augment"] = ["random_permute_features", "attribute_mask", "noise_then_mask"]
        cfg["augment_mode"] = _rand_choice(COARSE_SPACE["augment_mode"])
        # 增强强度
        cfg["noise_std"] = _rand_choice(COARSE_SPACE["noise_std"])
        cfg["mask_rate"] = _rand_choice(COARSE_SPACE["mask_rate"])
        cfgs.append(cfg)
    return cfgs


def build_refine_grid(top_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    围绕 top 配置构建小邻域网格（适度，避免爆炸）
    """
    lr = float(top_cfg.get("lr", 5e-4))
    dropout = float(top_cfg.get("dropout", 0.1))
    ed = int(top_cfg.get("dimensions", 128))
    lr_mults = [0.5, 0.8, 1.0, 1.25, 1.5]
    dropout_vals = sorted(set([0.0, 0.05, 0.1, 0.15, 0.2, max(0.0, round(dropout - 0.05, 2)), min(0.5, round(dropout + 0.05, 2))]))
    ed_near = sorted(set([max(16, ed // 2), ed, min(512, ed * 2)]))
    proj_opts = [None, top_cfg.get("proj_dim", None), max(16, int(top_cfg.get("hidden2", ed // 2)))]

    grid: List[Dict[str, Any]] = []
    for lm in lr_mults:
        for d in dropout_vals:
            for e2 in ed_near:
                for pd in proj_opts:
                    cfg = dict(top_cfg)
                    cfg["lr"] = lr * lm
                    cfg["dropout"] = float(d)
                    cfg["dimensions"] = int(e2)
                    cfg["proj_dim"] = None if pd in (None, 0) else int(pd)
                    # 轻微扰动 loss 比例
                    for k in ("loss_ratio1", "loss_ratio2", "loss_ratio3"):
                        v = float(cfg.get(k, 0.0))
                        cfg[k] = max(0.0, round(v, 3))
                    grid.append(cfg)
    # 去重（以核心键做哈希）
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for c in grid:
        key = (
            c["lr"], c["dropout"], c["dimensions"], c.get("proj_dim", None),
            c["loss_ratio1"], c["loss_ratio2"], c["loss_ratio3"]
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    # 限制数量，避免过大（例如最多 60）
    return uniq[:10]


def _cfg_to_args(task: str, cfg: Dict[str, Any], epochs: int, seed: int, run_name: str) -> List[str]:
    """
    将配置映射为 main.py 的 CLI 参数列表
    """
    args: List[str] = [
        f"--task_type={task}",
        f"--epochs={epochs}",
        f"--seed={seed}",
        f"--run_name={run_name}",
        f"--dimensions={int(cfg['dimensions'])}",
        f"--hidden1={int(cfg['hidden1'])}",
        f"--hidden2={int(cfg['hidden2'])}",
        f"--lr={float(cfg['lr'])}",
        f"--weight_decay={float(cfg['weight_decay'])}",
        f"--dropout={float(cfg['dropout'])}",
        f"--batch={int(cfg['batch'])}",
        f"--moco_queue={int(cfg['moco_queue'])}",
        f"--moco_momentum={float(cfg['moco_momentum'])}",
        f"--moco_t={float(cfg['moco_t'])}",
        f"--loss_ratio1={float(cfg['loss_ratio1'])}",
        f"--loss_ratio2={float(cfg['loss_ratio2'])}",
        f"--loss_ratio3={float(cfg['loss_ratio3'])}",
        # 统一与基线一致，强制使用 one_hot 特征以保持可比性
        f"--feature_type=one_hot",
        # 将增强列表转换为逗号分隔字符串传入
        f"--augment={','.join(cfg['augment'])}",
        f"--augment_mode={cfg.get('augment_mode', 'static')}"
    ]
    # proj_dim 可为 None
    if cfg.get("proj_dim", None) not in (None, 0):
        args.append(f"--proj_dim={int(cfg['proj_dim'])}")
    else:
        args.append(f"--proj_dim={int(cfg['hidden2'])}")
    # 可选：相似度阈值（若提供）
    if "similarity_threshold" in cfg:
        args.append(f"--similarity_threshold={float(cfg['similarity_threshold'])}")
    # 为固定三增强统一传入增强强度参数
    args.append(f"--noise_std={float(cfg.get('noise_std', 0.01))}")
    args.append(f"--mask_rate={float(cfg.get('mask_rate', 0.1))}")
    return args


def _record_experiment(task: str, run_name: str, cfg: Dict[str, Any], cmd: str, run_dir: Optional[Path]) -> None:
    """
    在 experiments/<task>/<run_name>/ 写 params.json 与 command.txt，并指向 OUTPUT 结果路径
    """
    exp_dir = EXP_DIR / f"{_ts()}_{task}"
    run_dir_local = exp_dir / run_name
    _ensure_dir(run_dir_local)
    _write_json(run_dir_local / "params.json", {
        "task": task,
        "run_name": run_name,
        "config": cfg,
        "command": cmd,
        "output_result_dir": str(run_dir) if run_dir else None
    })
    (run_dir_local / "command.txt").write_text(cmd, encoding="utf-8")


def run_baseline(task: str) -> Dict[str, Any]:
    print(f"========== Stage A | Baseline | Task={task} ==========")
    run_name = f"{task}_baseline"
    # 基线参数对齐为与用户给定命令等效（按任务选择文件）
    args = [
        f"--task_type={task}",
        "--validation_type=5-cv1",
        "--feature_type=one_hot",
        "--similarity_threshold=0.5",
        "--dimensions=64",
        "--lr=0.0005",
        "--weight_decay=0.0005",
        "--epochs=3",
        "--alpha=0.5",
        "--beta=0.5",
        "--gamma=0.5",
        f"--file=dataset1/{task}.edgelist",
        f"--neg_sample=dataset1/non_{task}.edgelist",
        f"--run_name={run_name}"
    ]
    rc, elapsed, cmd = _run_main(args)
    run_dir = _find_run_result_dir(run_name)
    m = _parse_result_summary(run_dir) if rc == 0 else {}
    _record_experiment(task, run_name, {"baseline": True}, cmd, run_dir)
    row = _trial_row(task, run_name, {"baseline": True}, m, elapsed, run_dir, cmd)
    print(f"[A] {task} | {json.dumps(row, ensure_ascii=False)}")
    return row


def run_random_search(task: str, N: int, epochs: int, base_seed: int = 0, epochs_cand: Optional[List[int]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    print(f"========== Stage B | Random Search | Task={task} | N={N}, epochs={epochs} ==========")
    history_rows: List[Dict[str, Any]] = []
    cfgs = sample_coarse_configs(N, task, base_seed=base_seed)
    hist_csv = OUTPUT_DIR / "result" / f"opt_history_{task}.csv"
    hist_jsonl = OUTPUT_DIR / "result" / f"opt_history_{task}.jsonl"
    _ensure_dir(hist_csv.parent)
    if hist_csv.exists():
        hist_csv.unlink(missing_ok=True)
    if hist_jsonl.exists():
        hist_jsonl.unlink(missing_ok=True)
    _append_line(hist_csv, "task,run_name,auprc_mean,auprc_std,auroc_mean,auroc_std,f1_mean,f1_std,elapsed_sec,result_dir,command")

    # 决定本阶段 epochs 候选
    _epochs_cand = list(epochs_cand) if epochs_cand else list(EPOCHS_CAND_DEFAULT)
    for idx, cfg in enumerate(cfgs, start=1):
        run_name = f"{task}_B_{idx:03d}"
        # 为每个 trial 采样 epochs（若提供候选），否则使用传入 epochs
        _ep = _rand_choice(_epochs_cand) if (_epochs_cand and len(_epochs_cand) > 0) else epochs
        cfg["epochs"] = int(_ep)
        args = _cfg_to_args(task, cfg, epochs=int(_ep), seed=base_seed, run_name=run_name)
        rc, elapsed, cmd = _run_main(args)
        if rc != 0:
            # 检测失败，尝试 lr * 0.5 重试一次
            try_cfg = dict(cfg)
            try_cfg["lr"] = float(cfg["lr"]) * 0.5
            print(f"[WARN] Trial failed. Retry with lr*0.5 = {try_cfg['lr']}")
            args = _cfg_to_args(task, try_cfg, epochs=epochs, seed=base_seed, run_name=run_name + "_retry")
            rc, elapsed, cmd = _run_main(args)
            cfg = try_cfg  # 以重试配置为准
            if rc != 0:
                # 记错误并继续
                err_file = OUTPUT_DIR / "result" / "error_report.txt"
                _append_line(err_file, f"{_ts()} | {task} | {run_name} failed twice. CMD: {cmd}")
        run_dir = _find_run_result_dir(run_name if rc == 0 else run_name + "_retry")
        m = _parse_result_summary(run_dir) if (run_dir and (rc == 0)) else {}
        _record_experiment(task, run_name, cfg, cmd, run_dir)
        row = _trial_row(task, run_name, cfg, m, elapsed, run_dir, cmd)
        history_rows.append(row)
        _append_line(hist_jsonl, json.dumps(row, ensure_ascii=False))
        _append_line(hist_csv, ",".join([
            task, run_name, str(row.get("auprc_mean")), str(row.get("auprc_std")),
            str(row.get("auroc_mean")), str(row.get("auroc_std")),
            str(row.get("f1_mean")), str(row.get("f1_std")),
            str(row.get("elapsed_sec")), str(row.get("result_dir") or ""), '"' + row.get("command", "").replace('"','""') + '"'
        ]))
        # 打印实时 top10
        valid = [r for r in history_rows if r.get("auprc_mean") is not None]
        valid.sort(key=lambda x: x["auprc_mean"], reverse=True)
        topk = valid[:10]
        print(f"[B-TOP] Task={task} current top10 by AUPRC:")
        for i, r in enumerate(topk, start=1):
            print(f"  {i:02d}. {r['run_name']} AUPRC={r.get('auprc_mean'):.4f}±{(r.get('auprc_std') or 0):.4f}")

    # 保存 top10
    valid = [r for r in history_rows if r.get("auprc_mean") is not None]
    valid.sort(key=lambda x: x["auprc_mean"], reverse=True)
    top10 = valid[:10]
    top10_csv = OUTPUT_DIR / "result" / f"configs_top10_task_{task}.csv"
    _ensure_dir(top10_csv.parent)
    _append_line(top10_csv, "rank,run_name,auprc_mean,auprc_std,auroc_mean,auroc_std,f1_mean,f1_std,config_json,result_dir")
    for i, r in enumerate(top10, start=1):
        _append_line(top10_csv, ",".join([
            str(i), r["run_name"], str(r.get("auprc_mean")), str(r.get("auprc_std")),
            str(r.get("auroc_mean")), str(r.get("auroc_std")),
            str(r.get("f1_mean")), str(r.get("f1_std")),
            json.dumps(r["config"], ensure_ascii=False).replace(",", ";"), str(r.get("result_dir") or "")
        ]))
    print(f"[B] Task={task} finished. top10 saved to {top10_csv}")
    return history_rows, top10


def run_refine(task: str, top10: List[Dict[str, Any]], epochs: int, base_seed: int = 0) -> List[Dict[str, Any]]:
    print(f"========== Stage C | Refine | Task={task} | epochs={epochs} ==========")
    refine_rows: List[Dict[str, Any]] = []
    # 仅用阶段B的 top2 作为精调中心，控制总命令量
    for rank, top in enumerate(top10[:2], start=1):
        cfg_center = top["config"]
        grid = build_refine_grid(cfg_center)
        for j, cfg in enumerate(grid, start=1):
            run_name = f"{task}_C_r{rank}_{j:03d}"
            args = _cfg_to_args(task, cfg, epochs=epochs, seed=base_seed, run_name=run_name)
            rc, elapsed, cmd = _run_main(args)
            if rc != 0:
                # 简单记录错误并继续（此处不再重复重试）
                err_file = OUTPUT_DIR / "result" / "error_report.txt"
                _append_line(err_file, f"{_ts()} | {task} | {run_name} failed. CMD: {cmd}")
            run_dir = _find_run_result_dir(run_name)
            m = _parse_result_summary(run_dir) if run_dir else {}
            _record_experiment(task, run_name, cfg, cmd, run_dir)
            row = _trial_row(task, run_name, cfg, m, elapsed, run_dir, cmd)
            refine_rows.append(row)
        # 每个中心打印一次该中心的 top3
        valid = [r for r in refine_rows if r.get("auprc_mean") is not None]
        valid.sort(key=lambda x: x["auprc_mean"], reverse=True)
        print(f"[C-PROG] Task={task} refine cumulative top5:")
        for i, r in enumerate(valid[:5], start=1):
            print(f"  {i:02d}. {r['run_name']} AUPRC={r.get('auprc_mean'):.4f}±{(r.get('auprc_std') or 0):.4f}")
    return refine_rows


def run_reproduce_top3(task: str, rows: List[Dict[str, Any]], seeds: List[int], full_epochs: int) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("auprc_mean") is not None]
    valid.sort(key=lambda x: x["auprc_mean"], reverse=True)
    top3 = valid[:3]
    finals: Dict[str, Any] = {"task": task, "finals": []}
    for i, r in enumerate(top3, start=1):
        cfg = r["config"]
        name_base = r["run_name"] + "_final"
        per_seed = []
        for sd in seeds:
            run_name = f"{name_base}_s{sd}"
            args = _cfg_to_args(task, cfg, epochs=full_epochs, seed=sd, run_name=run_name)
            rc, elapsed, cmd = _run_main(args)
            run_dir = _find_run_result_dir(run_name)
            m = _parse_result_summary(run_dir) if run_dir else {}
            _record_experiment(task, run_name, cfg, cmd, run_dir)
            per_seed.append({"seed": sd, "metrics": m, "result_dir": str(run_dir) if run_dir else None})
        # 聚合 seed 级（若 main 的5-fold均值已是 per-run 均值，这里仅统计 across seeds 的均值/方差）
        def _mean_std(vals):
            import statistics as S
            try:
                return float(S.mean(vals)), float(S.pstdev(vals))
            except Exception:
                return None, None
        auprc_vals = [x["metrics"].get("auprc_mean") for x in per_seed if x["metrics"].get("auprc_mean") is not None]
        auroc_vals = [x["metrics"].get("auroc_mean") for x in per_seed if x["metrics"].get("auroc_mean") is not None]
        f1_vals = [x["metrics"].get("f1_mean") for x in per_seed if x["metrics"].get("f1_mean") is not None]
        auprc_ms = _mean_std(auprc_vals) if auprc_vals else (None, None)
        auroc_ms = _mean_std(auroc_vals) if auroc_vals else (None, None)
        f1_ms = _mean_std(f1_vals) if f1_vals else (None, None)
        finals["finals"].append({
            "rank": i,
            "run_name": r["run_name"],
            "config": cfg,
            "per_seed": per_seed,
            "aggregate": {
                "auprc_mean": auprc_ms[0], "auprc_std": auprc_ms[1],
                "auroc_mean": auroc_ms[0], "auroc_std": auroc_ms[1],
                "f1_mean": f1_ms[0], "f1_std": f1_ms[1],
            }
        })
    # 写 best_configs_final.json
    out_json = OUTPUT_DIR / "result" / "best_configs_final.json"
    _write_json(out_json, finals)
    print(f"[FINAL] Task={task} best configs written: {out_json}")
    return finals


def write_summary_md(task: str, baseline_row: Dict[str, Any], history: List[Dict[str, Any]], refine_rows: List[Dict[str, Any]], finals: Dict[str, Any]) -> None:
    md = []
    md.append(f"# Summary for Task {task}")
    md.append("## Baseline")
    md.append("```json")
    md.append(json.dumps(baseline_row, ensure_ascii=False, indent=2))
    md.append("```")
    md.append("## Stage B Random Search (Top 10 by AUPRC)")
    valid = [r for r in history if r.get("auprc_mean") is not None]
    valid.sort(key=lambda x: x["auprc_mean"], reverse=True)
    for i, r in enumerate(valid[:10], start=1):
        md.append(f"- {i:02d}. {r['run_name']} AUPRC={r.get('auprc_mean')}")
    md.append("## Stage C Refine (Top 5)")
    valid2 = [r for r in refine_rows if r.get("auprc_mean") is not None]
    valid2.sort(key=lambda x: x["auprc_mean"], reverse=True)
    for i, r in enumerate(valid2[:5], start=1):
        md.append(f"- {i:02d}. {r['run_name']} AUPRC={r.get('auprc_mean')}")
    md.append("## Final Reproduction (Top3 x 3 seeds)")
    md.append("```json")
    md.append(json.dumps(finals, ensure_ascii=False, indent=2))
    md.append("```")
    out_md = OUTPUT_DIR / "result" / f"summary_task_{task}.md"
    _ensure_dir(out_md.parent)
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[SUMMARY] {out_md}")


def main_driver():
    import argparse
    p = argparse.ArgumentParser(description="A→B→C 自动化驱动（随机搜索+精调）")
    p.add_argument("--stage", type=str, default="auto", choices=["A", "B", "C", "auto"], help="执行阶段")
    p.add_argument("--tasks", type=str, default="LDA,MDA,LMI", help="任务列表，逗号分隔")
    p.add_argument("--num_random", type=int, default=20, help="阶段B随机搜索配置数量")
    p.add_argument("--epochs_b", type=int, default=3, help="阶段B每trial训练轮数")
    p.add_argument("--epochs_c", type=int, default=10, help="阶段C每trial训练轮数")
    p.add_argument("--full_epochs", type=int, default=50, help="最终复现的训练轮数")
    p.add_argument("--seeds", type=str, default="11,22,33", help="最终复现的种子，逗号分隔")
    p.add_argument("--base_seed", type=int, default=0, help="采样基准随机种子")
    # 新增：阶段B的 epochs 候选（逗号分隔），默认启用 3,50,100
    p.add_argument("--epochs_cand", type=str, default="3,50,100", help="阶段B随机搜索的 epochs 候选，逗号分隔")
    args = p.parse_args()

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("========== HPO Driver Start ==========")
    print(f"Tasks: {tasks}")
    print(f"Stage: {args.stage}")
    print(f"B: N={args.num_random}, epochs={args.epochs_b} | C: epochs={args.epochs_c} | Final epochs={args.full_epochs} seeds={seeds}")

    # 解析 epochs 候选列表
    try:
        epochs_cand = [int(x.strip()) for x in (args.epochs_cand or "").split(",") if x.strip()]
    except Exception:
        epochs_cand = EPOCHS_CAND_DEFAULT

    for task in tasks:
        baseline_row = None
        history: List[Dict[str, Any]] = []
        top10: List[Dict[str, Any]] = []
        refine_rows: List[Dict[str, Any]] = []
        finals: Dict[str, Any] = {}

        if args.stage in ("A", "auto"):
            baseline_row = run_baseline(task)

        if args.stage in ("B", "auto"):
            history, top10 = run_random_search(task, args.num_random, args.epochs_b, base_seed=args.base_seed, epochs_cand=epochs_cand)

        if args.stage in ("C", "auto"):
            # 若未运行B但 stage=C ，则从历史文件读取 top10
            if not top10:
                hist_jsonl = OUTPUT_DIR / "result" / f"opt_history_{task}.jsonl"
                if hist_jsonl.exists():
                    rows = [json.loads(x) for x in hist_jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
                    valid = [r for r in rows if r.get("auprc_mean") is not None]
                    valid.sort(key=lambda x: x["auprc_mean"], reverse=True)
                    top10 = valid[:10]
            refine_rows = run_refine(task, top10, args.epochs_c, base_seed=args.base_seed)
            # 跳过最终复现以控制总命令量
            finals = {}
            write_summary_md(task, baseline_row or {}, history, refine_rows, finals)

    print("========== HPO Driver End ==========")


if __name__ == "__main__":
    # 允许直接以 python hyperparameter-tuning/autodl.py 运行自动化驱动
    main_driver()