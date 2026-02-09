from __future__ import division  # 确保 '/' 执行浮点数除法
from __future__ import print_function  # 确保 print 是函数（兼容 Python 2/3）
import os  # 路径解析
import numpy as np  # 数值与数组计算
import scipy.sparse as sp  # 稀疏矩阵操作
import torch  # PyTorch
import torch.nn as nn  # 神经网络模块
from torch import Tensor  # 类型注解
from typing import Any, Optional, Tuple, List  # 类型注解
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
    根据输入的部分内容生成一个确定性的种子值
    
    该函数通过哈希算法将多个输入部分组合成一个32位整数种子，用于确保实验的可重现性。
    算法使用了一个类似于Python内置hash算法的参数（1000003）来混合输入值。
    
    Args:
        *parts: 可变数量的输入参数，每个参数都会被尝试转换为整数参与计算
        
    Returns:
        int: 生成的种子值（32位正整数），如果计算结果为0，则返回1以避免触发某些库的默认行为
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
    统一对抗训练种子派生函数
    规则：base = (adv_seed or seed) + fold，再派生 batch 级种子
    
    Args:
        args: 包含配置参数的对象，应包含 adv_seed 或 seed 属性
        fold: 折数索引，用于区分不同折的数据
        epoch: 训练轮次索引，默认为 0
        batch: 批次索引，默认为 0
        
    Returns:
        int: 派生出的种子值，可用于对抗训练中保证可重现性
    """
    try:
        # 尝试获取 adv_seed，如果不存在则使用 seed
        base = getattr(args, "adv_seed", None)
        base = int(base) if base is not None else int(getattr(args, "seed", 0))
    except Exception:
        # 出现异常时使用 seed 值
        base = int(getattr(args, "seed", 0))
    # 将基础种子与折数相加得到新的基础值
    base += int(fold or 0)
    # 使用 derive_seed 函数生成最终的种子值
    return derive_seed(base, epoch, batch)

# ========= 断言工具(用于验证代码逻辑) =========
def assert_tensor_2d(x: Tensor, name: str) -> None:
    """
    断言输入张量 x 是一个二维的 torch.Tensor
    
    Args:
        x (Tensor): 待检查的张量
        name (str): 张量的名称，用于错误提示信息
    
    Raises:
        TypeError: 当 x 不是 torch.Tensor 类型或不是二维张量时抛出异常
    """
    if not isinstance(x, torch.Tensor) or x.dim() != 2:
        raise TypeError(f"{name} must be a 2D torch.Tensor, got {type(x)} with shape {getattr(x, 'shape', None)}")

def assert_edge_index(edge_index: Tensor, name: str) -> None:
    """
    断言 edge_index 是一个形状为 [2, E] 且数据类型为整数的 torch.Tensor
    
    Args:
        edge_index (Tensor): 边索引张量，应该是一个形状为 [2, E] 的二维张量
        name (str): 张量的名称，用于错误提示信息
    
    Raises:
        TypeError: 当 edge_index 不是 torch.Tensor 类型或数据类型不是整数时抛出异常
        ValueError: 当 edge_index 不是二维张量或第一维大小不等于2时抛出异常
    """
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"{name} must have shape [2, E], got {tuple(edge_index.shape)}")
    if edge_index.dtype not in (torch.long, torch.int64, torch.int32, torch.int16):
        raise TypeError(f"{name} dtype must be integer type, got {edge_index.dtype}")

def assert_dense_adj(A: Tensor, N: int, name: str) -> None:
    """
    断言 A 是一个 N×N 的稠密邻接矩阵
    
    Args:
        A (Tensor): 待检查的邻接矩阵张量
        N (int): 邻接矩阵应有的行列数
        name (str): 张量的名称，用于错误提示信息
    
    Raises:
        ValueError: 当 A 不是 torch.Tensor 类型、不是二维张量或尺寸不匹配时抛出异常
    """
    if not isinstance(A, torch.Tensor) or A.dim() != 2 or A.size(0) != N or A.size(1) != N:
        raise ValueError(f"{name} must be dense square adj of shape [{N},{N}], got {getattr(A, 'shape', None)}")

# ========= 读出/初始化与特征增强（自 layer.py 迁移） =========
def reset_parameters(w: torch.Tensor):
    """
    权重参数初始化函数
    
    使用均匀分布初始化权重参数，范围取决于输入维度的平方根。
    分布范围为[-stdv, stdv]，其中stdv = 1.0 / sqrt(input_size)
    
    Args:
        w (torch.Tensor): 待初始化的权重张量
    """
    stdv = 1.0 / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)

class AvgReadout(nn.Module):
    """
    节点表示的图级汇聚模块
    
    支持带掩码的平均读出操作，用于将节点序列表示聚合为图级表示。
    如果提供了掩码，则只对未被掩码的节点进行平均；否则对所有节点进行平均。
    """
    
    def __init__(self):
        """
        初始化AvgReadout模块
        """
        super().__init__()
        
    def forward(self, seq, msk=None):
        """
        前向传播函数
        
        Args:
            seq (torch.Tensor): 输入序列张量
            msk (torch.Tensor, optional): 掩码张量，用于标识有效节点
            
        Returns:
            torch.Tensor: 聚合后的图级表示
        """
        if msk is None:
            return torch.mean(seq, 0)
        msk = torch.unsqueeze(msk, -1)
        return torch.sum(seq * msk, 0) / torch.sum(msk)


def _make_generator(seed: Optional[int], device: torch.device) -> Optional[torch.Generator]:
    """
    创建本地随机数发生器
    
    创建一个独立的随机数生成器，避免污染全局随机状态。
    
    Args:
        seed (Optional[int]): 随机种子，如果为None则返回None
        device (torch.device): 设备类型
        
    Returns:
        Optional[torch.Generator]: 随机数生成器或None
    """
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g

def random_permute_features(X: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
    """
    随机重排特征矩阵的行顺序
    
    按样本维度随机重排行顺序，仅支持二维张量。
    
    Args:
        X (torch.Tensor): 输入的二维特征矩阵
        seed (Optional[int]): 随机种子
        
    Returns:
        torch.Tensor: 行顺序被打乱的特征矩阵
        
    Raises:
        ValueError: 当输入不是二维张量时抛出异常
    """
    if not isinstance(X, torch.Tensor) or X.dim() != 2:
        raise ValueError("random_permute_features 仅支持 2D Tensor")
    N = X.size(0)
    g = _make_generator(seed, X.device)
    idx = torch.randperm(N, device=X.device, generator=g) if N > 0 else torch.empty(0, dtype=torch.long, device=X.device)
    return X.index_select(0, idx)

def add_noise(X: torch.Tensor, noise_std: float = 0.01, seed: Optional[int] = None) -> torch.Tensor:
    """
    向特征矩阵添加高斯噪声
    
    添加零均值、指定标准差的高斯噪声到输入特征矩阵。
    
    Args:
        X (torch.Tensor): 输入的二维特征矩阵
        noise_std (float): 噪声的标准差，默认为0.01
        seed (Optional[int]): 随机种子
        
    Returns:
        torch.Tensor: 添加噪声后的特征矩阵
        
    Raises:
        ValueError: 当输入不是二维张量时抛出异常
    """
    if noise_std <= 0:
        return X
    if not isinstance(X, torch.Tensor) or X.dim() != 2:
        raise ValueError("add_noise 仅支持 2D Tensor")
    g = _make_generator(seed, X.device)
    noise = torch.randn(X.size(), device=X.device, dtype=X.dtype, generator=g) * float(noise_std)
    return X + noise

def attribute_mask(X: torch.Tensor, mask_rate: float = 0.1, seed: Optional[int] = None) -> torch.Tensor:
    """
    对特征矩阵进行属性掩码
    
    以列为单位随机选择部分特征维度进行掩码（置零）处理。
    
    Args:
        X (torch.Tensor): 输入的二维特征矩阵
        mask_rate (float): 掩码比例，默认为0.1（10%的特征维度会被置零）
        seed (Optional[int]): 随机种子
        
    Returns:
        torch.Tensor: 经过属性掩码处理的特征矩阵
        
    Raises:
        ValueError: 当输入不是二维张量时抛出异常
    """
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
    """
    先加噪声再进行属性掩码的组合增强方法
    
    依次应用添加噪声和属性掩码两种增强操作，两个子步骤使用不同的随机种子。
    
    Args:
        X (torch.Tensor): 输入的二维特征矩阵
        noise_std (float): 噪声标准差，默认为0.01
        mask_rate (float): 掩码比例，默认为0.1
        seed (Optional[int]): 基础随机种子
        
    Returns:
        torch.Tensor: 经过组合增强处理的特征矩阵
    """
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
    """
    应用指定的数据增强方法
    
    根据名称调度相应的数据增强策略，支持多种增强方法。
    
    Args:
        name (str): 增强方法名称
        X (torch.Tensor): 输入的特征张量
        noise_std (float): 噪声标准差，默认为0.01
        mask_rate (float): 掩码比例，默认为0.1
        seed (Optional[int]): 随机种子
        
    Returns:
        torch.Tensor: 经过增强处理的特征张量
        
    Raises:
        ValueError: 当增强方法名称未知或输入不符合要求时抛出异常
    """
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(X, torch.Tensor) or X.dim() != 2:
        raise ValueError(f"apply_augmentation 要求 2D Tensor，得到 {type(X)} shape={getattr(X, 'shape', None)}")
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
    """
    对稀疏矩阵执行按行归一化（Row-normalize）
    [将矩阵每一行的数据转换到统一的尺度上]
    
    Args:
        mx: 输入的稀疏矩阵
        
    Returns:
        归一化后的稀疏矩阵
    """
    rowsum = np.array(mx.sum(1))  # 每行求和
    r_inv = np.power(rowsum, -1).flatten()  # 行和的倒数
    r_inv[np.isinf(r_inv)] = 0.  # 将 inf 替换为 0（行和为 0 的行）
    r_mat_inv = sp.diags(r_inv)  # 对角矩阵
    mx = r_mat_inv.dot(mx)  # 左乘实现按行缩放
    return mx

def Preproces_Data (A, test_id):
    """
    在关联矩阵 A 中将测试集中的已知关联置 0（构造训练视图）
    
    Args:
        A: 原始关联矩阵
        test_id: 测试集样本的索引数组
        
    Returns:
        处理后的关联矩阵（测试集关联置0）
    """
    copy_A = A / 1  # 浅复制矩阵，避免改动原始数据
    for i in range(test_id.shape[0]):  # 遍历测试样本 ID
        copy_A[int(test_id[i][0])][int(test_id[i][1])] = 0
    return copy_A

def construct_graph(lncRNA_disease,  miRNA_disease, miRNA_lncRNA, lncRNA_sim, miRNA_sim, disease_sim):
    """
    构建包含 lncRNA / disease / miRNA 的异构图邻接矩阵
    
    Args:
        lncRNA_disease: lncRNA-disease 关联矩阵
        miRNA_disease: miRNA-disease 关联矩阵
        miRNA_lncRNA: miRNA-lncRNA 关联矩阵
        lncRNA_sim: lncRNA 相似性矩阵
        miRNA_sim: miRNA 相似性矩阵
        disease_sim: disease 相似性矩阵
        
    Returns:
        完整的异构图邻接矩阵
    """
    # lncRNA 视角：[lncRNA-相似度, lncRNA-disease 关联, lncRNA-miRNA 关联]
    lnc_dis_sim = np.hstack((lncRNA_sim, lncRNA_disease, miRNA_lncRNA.T))
    # disease 视角：[disease-lncRNA 关联, disease-相似度, disease-miRNA 关联]
    dis_lnc_sim = np.hstack((lncRNA_disease.T, disease_sim, miRNA_disease.T))
    # miRNA 视角：[miRNA-lncRNA 关联, miRNA-disease 关联, miRNA-相似度]
    mi_lnc_dis = np.hstack((miRNA_lncRNA, miRNA_disease, miRNA_sim))
    # 拼接为整体异构图邻接矩阵
    matrix_A = np.vstack((lnc_dis_sim, dis_lnc_sim, mi_lnc_dis))
    return matrix_A

def lalacians_norm(adj):
    """
    对邻接矩阵执行对称拉普拉斯归一化：D^(-0.5) A D^(-0.5)
    
    Args:
        adj: 输入的邻接矩阵
        
    Returns:
        对称拉普拉斯归一化后的矩阵
    """
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
    """
    数值安全的符号函数（零梯度时返回 0）
    
    该函数是对torch.sign的封装，用于在对抗训练中计算符号而不产生数值不稳定问题
    
    Args:
        x (Tensor): 输入张量
        
    Returns:
        Tensor: 返回与输入相同形状的符号张量，正值为1，负值为-1，零值为0
    """
    return torch.sign(x)

def l2_normalize(x: Tensor, eps: float = 1e-12) -> Tensor:
    """
    按 L2 范数对向量进行归一化
    
    将输入张量按L2范数进行归一化处理，即让向量的模长变为1，常用于计算方向向量
    
    Args:
        x (Tensor): 待归一化的输入张量
        eps (float): 防止除零错误的小量，默认为1e-12
        
    Returns:
        Tensor: 归一化后的张量，具有相同的形状
    """
    return x / (x.norm(p=2) + eps)

def project_to_ball(delta: Tensor, norm: str, eps: float) -> Tensor:
    """
    将增量 delta 投影回指定范数球
    
    根据指定的范数类型（linf或l2），将扰动增量投影到对应的范数球内，控制扰动幅度
    
    Args:
        delta (Tensor): 扰动增量张量
        norm (str): 范数类型，支持 "linf"（无穷范数）或 "l2"（二范数）
        eps (float): 范数球半径，即扰动的最大允许幅度
        
    Returns:
        Tensor: 投影后的扰动张量，满足指定范数约束
    """
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
    """
    根据范数类型执行一步未投影更新
    
    根据指定的范数类型，沿梯度方向更新扰动量，这是对抗攻击迭代过程中的核心步骤
    
    Args:
        delta (Tensor): 当前扰动量
        g (Tensor): 梯度张量，指示更新方向
        norm (str): 范数类型，支持 "linf"、"l2" 或其他范数
        alpha (float): 更新步长
        
    Returns:
        Tensor: 更新后的扰动量
    """
    if norm == "linf":
        return delta + alpha * sign_safe(g)
    elif norm == "l2":
        dir_vec = l2_normalize(g)
        return delta + alpha * dir_vec
    else:
        return delta + alpha * g

def clamp_features(x: Tensor, clip_min: float, clip_max: float) -> Tensor:
    """
    对扰动后的特征执行数值裁剪；允许 ±inf 作为无界
    
    将扰动后的特征值限制在指定范围内，防止超出合理的数值边界
    
    Args:
        x (Tensor): 扰动后的特征张量
        clip_min (float): 允许的最小值，可为 -inf 表示无下界
        clip_max (float): 允许的最大值，可为 +inf 表示无上界
        
    Returns:
        Tensor: 裁剪后的特征张量
    """
    if clip_min == float("-inf") and clip_max == float("inf"):
        return x
    return torch.clamp(x, min=clip_min, max=clip_max)

def maybe_rand_init_like(x: Tensor, norm: str, eps: float) -> Tensor:
    """
    按范数对增量进行随机初始化
    
    根据指定的范数类型和扰动幅度，随机初始化扰动增量张量
    
    Args:
        x (Tensor): 参考张量，用于确定输出的形状和设备
        norm (str): 范数类型，支持 "linf" 或 "l2"
        eps (float): 扰动幅度范围
        
    Returns:
        Tensor: 随机初始化的扰动增量张量
    """
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
    """
    判断对象是否为 torch.Tensor 类型
    
    Args:
        x (Any): 待判断的对象
        
    Returns:
        bool: 如果对象是 torch.Tensor 类型则返回 True，否则返回 False
    """
    """判断对象是否为 torch.Tensor"""
    return isinstance(x, torch.Tensor)

def _to_numpy(x: Any) -> Tuple[np.ndarray, Optional[Any]]:
    """
    将输入转换为 NumPy 数组，并返回 (np_array, like)。
    like 为原对象，用于后续从 numpy 还原原始类型。
    
    Args:
        x (Any): 待转换的对象，支持 torch.Tensor 或 numpy.ndarray 类型
        
    Returns:
        Tuple[np.ndarray, Optional[Any]]: 返回元组，第一个元素是转换后的 numpy 数组，
                                         第二个元素是原始对象用于类型恢复
        
    Raises:
        TypeError: 当输入既不是 torch.Tensor 也不是 numpy.ndarray 时抛出异常
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
    
    Args:
        x_np (np.ndarray): 待转换的 numpy 数组
        like (Any): 参考对象，用于确定目标类型
        
    Returns:
        Any: 转换后的对象，类型与参考对象相同
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
    """
    创建独立的 NumPy 随机数发生器（可选固定种子）
    
    Args:
        seed (Optional[int]): 随机种子，如果为 None 则使用随机种子
        
    Returns:
        np.random.Generator: NumPy 随机数生成器实例
    """
    """创建独立的 NumPy 随机数发生器（可选固定种子）"""
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()


# ========= BYOL 损失函数相关 =========
class BYOLLoss(nn.Module):
    """
    BYOL (Bootstrap Your Own Latent) 多视图对称损失函数
    基于 Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning 论文实现
    """
    
    def __init__(self, temperature: float = 0.2):
        """
        初始化BYOL损失函数
        
        Args:
            temperature: 温度系数，用于调整对比学习的强度
        """
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
    
    def forward(self, online_view1: torch.Tensor, online_view2: torch.Tensor,
                target_view1: torch.Tensor, target_view2: torch.Tensor) -> torch.Tensor:
        """
        计算BYOL对称损失
        
        Args:
            online_view1: 在线网络第一个视图的输出
            online_view2: 在线网络第二个视图的输出
            target_view1: 目标网络第一个视图的输出
            target_view2: 目标网络第二个视图的输出
            
        Returns:
            torch.Tensor: 对称BYOL损失值
        """
        # 确保所有输入张量形状一致
        assert online_view1.shape == online_view2.shape == target_view1.shape == target_view2.shape
        
        # 计算对称损失：L = 2 - 2 * (q1·z2 + q2·z1) / (||q1||·||z2|| + ||q2||·||z1||)
        
        # 视图1到视图2的损失
        loss_1_to_2 = 2 - 2 * self.cosine_similarity(online_view1, target_view2.detach()).mean()
        
        # 视图2到视图1的损失
        loss_2_to_1 = 2 - 2 * self.cosine_similarity(online_view2, target_view1.detach()).mean()
        
        # 对称损失的平均值
        symmetric_loss = (loss_1_to_2 + loss_2_to_1) / 2
        
        return symmetric_loss


def compute_byol_loss(predictions: List[torch.Tensor], targets: List[torch.Tensor], 
                     temperature: float = 0.2) -> torch.Tensor:
    """
    计算多视图BYOL损失（支持多个视图对）
    
    Args:
        predictions: 在线网络的预测输出列表
        targets: 目标网络的输出列表
        temperature: 温度系数
        
    Returns:
        torch.Tensor: 多视图BYOL损失的平均值
    """
    assert len(predictions) == len(targets), "预测和目标视图数量必须相同"
    
    byol_loss = BYOLLoss(temperature=temperature)
    
    # 计算所有视图对之间的损失
    losses = []
    num_views = len(predictions)
    
    for i in range(num_views):
        for j in range(num_views):
            if i != j:  # 避免相同视图的比较
                loss = byol_loss(predictions[i], predictions[j], 
                               targets[i].detach(), targets[j].detach())
                losses.append(loss)
    
    # 计算所有视图对损失的平均值
    if losses:
        return torch.stack(losses).mean()
    else:
        return torch.tensor(0.0, device=predictions[0].device)


def normalize_byol_features(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    对BYOL特征进行L2归一化
    
    Args:
        features: 输入特征张量
        eps: 数值稳定性常数
        
    Returns:
        torch.Tensor: L2归一化后的特征
    """
    return features / (features.norm(dim=1, keepdim=True) + eps)