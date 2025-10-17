from __future__ import division  # 确保 '/' 执行浮点数除法
from __future__ import print_function  # 确保 print 是函数（兼容 Python 2/3）
import os  # 路径解析
import numpy as np  # 数值与数组计算
import scipy.sparse as sp  # 稀疏矩阵操作
import torch  # PyTorch
import torch.nn as nn  # 神经网络模块
from torch import Tensor  # 类型注解
from typing import Any, Optional, Tuple  # 类型注解
import math  # 数学运算
BASE_DIR = os.path.dirname(__file__)

def em_path(path: str) -> str:
    """将相对路径解析为相对于 EM 目录的绝对路径；绝对路径则原样返回"""
    try:
        return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
    except Exception:
        # 兜底：异常时直接返回原始字符串
        return path

# 兼容性别名：允许 import _p 直接使用
_p = em_path

# ========= 统一随机性工具 =========
def set_global_seed(seed: int = 0, deterministic: bool = True):
    """
    统一设置 Python/NumPy/PyTorch/CUDA/cuDNN 的随机种子与确定性选项。
    - 设置 PYTHONHASHSEED
    - random/np.random/torch.manual_seed
    - CUDA: torch.cuda.manual_seed_all
    - cuDNN: deterministic=True, benchmark=False（避免非确定性）
    """
    import os
    import random
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    try:
        random.seed(int(seed))
    except Exception:
        pass
    try:
        import numpy as _np
        _np.random.seed(int(seed))
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(int(seed))
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(int(seed))
        if deterministic:
            try:
                _torch.backends.cudnn.deterministic = True
                _torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    except Exception:
        pass

def derive_seed(*parts) -> int:
    """
    从多个整型部分派生稳定的 32-bit 种子（不影响全局状态）。
    规则：FNV 风格线性混合并对 2**32 取模，避免溢出。
    常用模式：
      - 每折：derive_seed(base_seed, fold)
      - 每 epoch/batch：derive_seed(base_seed, epoch, iter)
    ✅ 修复2：统一对抗种子派生，避免重复逻辑
    """
    mod = 2**32
    acc = 0
    for p in parts:
        try:
            v = int(p)
        except Exception:
            v = 0
        acc = (acc * 1000003 + v) % mod
    # 避免 0 触发部分库的默认非确定性路径
    return int(acc if acc != 0 else 1)

def derive_adv_seed(args: object, fold: int, epoch: int = 0, batch: int = 0) -> int:
    """
    ✅ 修复2：统一对抗训练种子派生函数
    规则：base = (adv_seed or seed) + fold，再派生 batch 级种子
    """
    try:
        base = getattr(args, "adv_seed", None)
        base = int(base) if base is not None else int(getattr(args, "seed", 0))
    except Exception:
        base = int(getattr(args, "seed", 0))
    base += int(fold or 0)
    return derive_seed(base, epoch, batch)

# ========= 断言工具 =========
def assert_tensor_2d(x: Tensor, name: str) -> None:
    """断言 x 为 2D torch.Tensor"""
    if not isinstance(x, torch.Tensor) or x.dim() != 2:
        raise TypeError(f"{name} must be a 2D torch.Tensor, got {type(x)} with shape {getattr(x, 'shape', None)}")

def assert_edge_index(edge_index: Tensor, name: str) -> None:
    """断言 edge_index 形状为 [2, E] 且整型"""
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"{name} must have shape [2, E], got {tuple(edge_index.shape)}")
    if edge_index.dtype not in (torch.long, torch.int64, torch.int32, torch.int16):
        raise TypeError(f"{name} dtype must be integer type, got {edge_index.dtype}")

def assert_dense_adj(A: Tensor, N: int, name: str) -> None:
    """断言 A 为 N×N 的稠密邻接矩阵"""
    if not isinstance(A, torch.Tensor) or A.dim() != 2 or A.size(0) != N or A.size(1) != N:
        raise ValueError(f"{name} must be dense square adj of shape [{N},{N}], got {getattr(A, 'shape', None)}")

# ========= 读出/初始化与特征增强（自 layer.py 迁移） =========
def reset_parameters(w: torch.Tensor):
    # 权重参数初始化：均匀分布，范围依赖输入维度
    stdv = 1.0 / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)

class AvgReadout(nn.Module):
    # 节点表示的图级汇聚：支持带掩码的平均读出
    def __init__(self):
        super().__init__()
    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        msk = torch.unsqueeze(msk, -1)
        return torch.sum(seq * msk, 0) / torch.sum(msk)

# 轻量增强：仅针对二维特征矩阵 X，保持与当前训练使用一致
def _make_generator(seed: Optional[int], device: torch.device) -> Optional[torch.Generator]:
    # 创建本地随机数发生器，避免污染全局状态
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g

def random_permute_features(X: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
    # 按样本维随机重排行顺序
    if not isinstance(X, torch.Tensor) or X.dim() != 2:
        raise ValueError("random_permute_features 仅支持 2D Tensor")
    N = X.size(0)
    g = _make_generator(seed, X.device)
    idx = torch.randperm(N, device=X.device, generator=g) if N > 0 else torch.empty(0, dtype=torch.long, device=X.device)
    return X.index_select(0, idx)

def add_noise(X: torch.Tensor, noise_std: float = 0.01, seed: Optional[int] = None) -> torch.Tensor:
    # 添加零均值高斯噪声
    if noise_std <= 0:
        return X
    if not isinstance(X, torch.Tensor) or X.dim() != 2:
        raise ValueError("add_noise 仅支持 2D Tensor")
    g = _make_generator(seed, X.device)
    noise = torch.randn(X.size(), device=X.device, dtype=X.dtype, generator=g) * float(noise_std)
    return X + noise

def attribute_mask(X: torch.Tensor, mask_rate: float = 0.1, seed: Optional[int] = None) -> torch.Tensor:
    # 以列为单位进行特征掩蔽（置零）
    if mask_rate <= 0:
        return X
    if not isinstance(X, torch.Tensor) or X.dim() != 2:
        raise ValueError("attribute_mask 仅支持 2D Tensor")
    N, D = X.size()
    k = int(float(mask_rate) * D)
    if k <= 0:
        return X
    k = min(k, D)
    g = _make_generator(seed, X.device)
    cols = torch.randperm(D, device=X.device, generator=g)[:k]
    out = X.clone()
    out[:, cols] = 0
    return out

def noise_then_mask(X: torch.Tensor, noise_std: float = 0.01, mask_rate: float = 0.1, seed: Optional[int] = None) -> torch.Tensor:
    # 先加噪声再掩蔽，两个子步骤可复用同一基准随机种子
    base = int(seed) if seed is not None else None
    x1 = add_noise(X, noise_std=noise_std, seed=base)
    x2 = attribute_mask(x1, mask_rate=mask_rate, seed=None if base is None else base + 1)
    return x2

def apply_augmentation(
    name: str,
    X: torch.Tensor,
    *,
    noise_std: float = 0.01,
    mask_rate: float = 0.1,
    seed: Optional[int] = None,
) -> torch.Tensor:
    # ✅ 修复3：兼容 numpy.ndarray 输入，自动转换为 torch.Tensor
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(X, torch.Tensor) or X.dim() != 2:
        raise ValueError(f"apply_augmentation 要求 2D Tensor，得到 {type(X)} shape={getattr(X, 'shape', None)}")
    # 根据名称调度增强策略；保持与历史名称兼容
    key = (name or "").strip().lower()
    if key in ("random_permute_features",):
        return random_permute_features(X, seed=seed)
    if key in ("add_noise",):
        return add_noise(X, noise_std=noise_std, seed=seed)
    if key in ("attribute_mask",):
        return attribute_mask(X, mask_rate=mask_rate, seed=seed)
    if key in ("noise_then_mask",):
        return noise_then_mask(X, noise_std=noise_std, mask_rate=mask_rate, seed=seed)
    if key in ("none", "null", ""):
        return X
    # 兼容原名未小写情况
    if name == "random_permute_features":
        return random_permute_features(X, seed=seed)
    if name == "attribute_mask":
        return attribute_mask(X, mask_rate=mask_rate, seed=seed)
    if name == "noise_then_mask":
        return noise_then_mask(X, noise_std=noise_std, mask_rate=mask_rate, seed=seed)
    if name == "add_noise":
        return add_noise(X, noise_std=noise_std, seed=seed)
    raise ValueError(f"Unknown augmentation name: {name}")

# ========= 稀疏/邻接矩阵相关 =========
def normalize(mx):
    """对稀疏矩阵执行按行归一化（Row-normalize）"""
    rowsum = np.array(mx.sum(1))  # 每行求和
    r_inv = np.power(rowsum, -1).flatten()  # 行和的倒数
    r_inv[np.isinf(r_inv)] = 0.  # 将 inf 替换为 0（行和为 0 的行）
    r_mat_inv = sp.diags(r_inv)  # 对角矩阵
    mx = r_mat_inv.dot(mx)  # 左乘实现按行缩放
    return mx

def Preproces_Data (A, test_id):
    """在关联矩阵 A 中将测试集中的已知关联置 0（构造训练视图）"""
    copy_A = A / 1  # 浅复制矩阵，避免改动原始数据
    for i in range(test_id.shape[0]):  # 遍历测试样本 ID
        copy_A[int(test_id[i][0])][int(test_id[i][1])] = 0
    return copy_A

'''构建包含 lncRNA / disease / miRNA 的异构图邻接矩阵'''
def construct_graph(lncRNA_disease,  miRNA_disease, miRNA_lncRNA, lncRNA_sim, miRNA_sim, disease_sim):
    # lncRNA 视角：[lncRNA-相似度, lncRNA-disease 关联, lncRNA-miRNA 关联]
    lnc_dis_sim = np.hstack((lncRNA_sim, lncRNA_disease, miRNA_lncRNA.T))
    # disease 视角：[disease-lncRNA 关联, disease-相似度, disease-miRNA 关联]
    dis_lnc_sim = np.hstack((lncRNA_disease.T, disease_sim, miRNA_disease.T))
    # miRNA 视角：[miRNA-lncRNA 关联, miRNA-disease 关联, miRNA-相似度]
    mi_lnc_dis = np.hstack((miRNA_lncRNA, miRNA_disease, miRNA_sim))
    # 拼接为整体异构图邻接矩阵
    matrix_A = np.vstack((lnc_dis_sim, dis_lnc_sim, mi_lnc_dis))
    return matrix_A

'''拉普拉斯对称归一化'''
def lalacians_norm(adj):
    """对邻接矩阵执行对称拉普拉斯归一化：D^(-0.5) A D^(-0.5)"""
    # adj += np.eye(adj.shape[0])  # 可选：添加自环（此处未使用）
    degree = np.array(adj.sum(1))  # 度（每行和）
    D = []  # 存储度的 -0.5 次方
    for i in range(len(degree)):
        if degree[i] != 0:
            de = np.power(degree[i], -0.5)
            D.append(de)
        else:
            D.append(0)
    degree = np.diag(np.array(D))  # D^(-0.5)
    norm_A = degree.dot(adj).dot(degree)  # 对称归一化
    # norm_A = degree.dot(adj)  # 左归一化（备用）
    return norm_A

# ========= 对抗扰动通用工具（供外部模块复用） =========
def sign_safe(x: Tensor) -> Tensor:
    """数值安全的符号函数（零梯度时返回 0）"""
    return torch.sign(x)

def l2_normalize(x: Tensor, eps: float = 1e-12) -> Tensor:
    """按 L2 范数对向量进行归一化"""
    return x / (x.norm(p=2) + eps)

def project_to_ball(delta: Tensor, norm: str, eps: float) -> Tensor:
    """将增量 delta 投影回指定范数球"""
    if eps <= 0:
        return torch.zeros_like(delta)
    if norm == "linf":
        return torch.clamp(delta, -eps, eps)
    elif norm == "l2":
        flat = delta.view(delta.size(0), -1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        factors = torch.minimum(torch.ones_like(norms), eps / norms)
        flat = flat * factors
        return flat.view_as(delta)
    else:
        return delta

def step_update(delta: Tensor, g: Tensor, norm: str, alpha: float) -> Tensor:
    """根据范数类型执行一步未投影更新"""
    if norm == "linf":
        return delta + alpha * sign_safe(g)
    elif norm == "l2":
        dir_vec = l2_normalize(g)
        return delta + alpha * dir_vec
    else:
        return delta + alpha * g

def clamp_features(x: Tensor, clip_min: float, clip_max: float) -> Tensor:
    """对扰动后的特征执行数值裁剪；允许 ±inf 作为无界"""
    if clip_min == float("-inf") and clip_max == float("inf"):
        return x
    return torch.clamp(x, min=clip_min, max=clip_max)

def maybe_rand_init_like(x: Tensor, norm: str, eps: float) -> Tensor:
    """按范数对增量进行随机初始化"""
    if eps <= 0:
        return torch.zeros_like(x)
    if norm == "linf":
        return torch.empty_like(x).uniform_(-eps, eps)
    elif norm == "l2":
        rand = torch.randn_like(x)
        rand = project_to_ball(rand, "l2", eps)
        return rand
    else:
        return torch.zeros_like(x)

# ========= enhance 通用工具（类型转换与 RNG） =========
def _is_torch_tensor(x: Any) -> bool:
    """判断对象是否为 torch.Tensor"""
    return isinstance(x, torch.Tensor)

def _to_numpy(x: Any) -> Tuple[np.ndarray, Optional[Any]]:
    """
    将输入转换为 NumPy 数组，并返回 (np_array, like)。
    like 为原对象，用于后续从 numpy 还原原始类型。
    """
    if _is_torch_tensor(x):
        # 确保在 CPU 且分离计算图；假设为稠密张量
        return x.detach().cpu().numpy(), x
    elif isinstance(x, np.ndarray):
        return x, x
    else:
        raise TypeError("Unsupported feature type. Expect numpy.ndarray or torch.Tensor.")

def _from_numpy_like(x_np: np.ndarray, like: Any) -> Any:
    """
    将 NumPy 数组还原为与 like 相同的类型。
    """
    if _is_torch_tensor(like):
        # 尽量保持原始 dtype 一致
        dtype = like.dtype if like is not None else None
        t = torch.from_numpy(x_np)
        if dtype is not None and t.dtype != dtype:
            try:
                t = t.to(dtype)
            except Exception:
                pass
        return t
    elif isinstance(like, np.ndarray):
        return x_np
    else:
        # 兜底：返回 numpy
        return x_np

def _rng(seed: Optional[int]) -> np.random.Generator:
    """创建独立的 NumPy 随机数发生器（可选固定种子）"""
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()