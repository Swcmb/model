"""
模型实例化模块

本模块负责构建深度学习模型与优化器，当前固定使用内聚版EM模型（包含encoder=gat_gt_serial, fusion=gt_fusion, MoCo多视图机制）。
模块主要提供模型创建、参数初始化和优化器配置功能。
"""
# 导入PyTorch优化器
from torch.optim import Adam
# 导入PyTorch用于类型判断
import torch  # 初始化器需要类型判断
# 从layer模块导入EM模型类
from layer import EM  # 绝对导入内聚版EM
# 从utils模块
# 导入权重初始化函数
from utils import reset_parameters  # 统一权重初始化


def _init_module(m):
    """
    模型参数统一初始化函数
    
    该函数仅对包含weight属性的模块进行参数初始化，使用reset_parameters函数
    实现权重的规范化初始化，提高模型训练稳定性。
    
    参数:
        m: torch.nn.Module - 需要初始化的PyTorch模块
    """
    try:
        # 检查模块是否有weight属性，无则设为None
        w = m.weight if hasattr(m, 'weight') else None
        # 仅当weight是PyTorch张量时进行初始化
        if isinstance(w, torch.Tensor):
            reset_parameters(w)
    except Exception:
        # 异常处理：单个模块初始化失败不影响整体模型初始化过程
        pass


def Create_model(args):
    """
    创建模型与优化器
    
    该函数根据输入参数构建EM模型，对模型参数进行初始化，并配置Adam优化器。
    当前实现固定使用内聚版EM模型架构，包含特定的编码器、融合器和多视图机制。
    
    参数:
        args: 包含模型配置的参数对象，必须包含以下属性：
            - dimensions: 输入特征维度
            - hidden1: 第一隐藏层维度
            - hidden2: 第二隐藏层维度
            - decoder1: 解码器第一层维度
            - dropout: Dropout概率值
            - lr: 学习率
            - weight_decay: 权重衰减系数
    
    返回:
        tuple: (model, optimizer)
            - model: 初始化后的EM模型实例
            - optimizer: 配置好的Adam优化器实例
    """
    # 固定使用内聚版EM模型（encoder=gat_gt_serial, fusion=gt_fusion, MoCo多视图）
    model = EM(
        feature=args.dimensions,      # 输入特征维度
        hidden1=args.hidden1,         # 第一隐藏层维度
        hidden2=args.hidden2,         # 第二隐藏层维度
        decoder1=args.decoder1,       # 解码器第一层维度
        dropout=args.dropout          # Dropout概率，用于正则化
    )
    
    # 模型参数统一初始化：递归应用_init_module函数对所有子模块进行初始化
    # 仅对包含weight属性的模块应用reset_parameters初始化
    try:
        model.apply(_init_module)
    except Exception:
        # 异常处理：初始化失败时不影响后续流程，保证程序健壮性
        pass
    
    # 筛选可训练参数：仅优化requires_grad为True的参数
    # 显式排除不需要梯度更新的参数，例如MoCo中的动量分支/缓冲
    trainable_params = (p for p in model.parameters() if getattr(p, "requires_grad", True))
    
    # 创建Adam优化器，配置学习率和权重衰减
    optimizer = Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # 返回初始化好的模型和优化器
    return model, optimizer