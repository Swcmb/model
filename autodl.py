import os
import platform
from typing import List, Optional

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