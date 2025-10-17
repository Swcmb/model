"""构建模型与优化器（已切换为使用 model.layer.EM）"""
from torch.optim import Adam
import torch  # 初始化器需要类型判断
from layer import EM  # 绝对导入内聚版 EM
from utils import reset_parameters  # 统一权重初始化

def _init_module(m):
    """统一初始化：仅对包含 weight 的模块进行 reset_parameters"""
    try:
        w = m.weight if hasattr(m, 'weight') else None
        if isinstance(w, torch.Tensor):
            reset_parameters(w)
    except Exception:
        # 单模块初始化失败不影响整体
        pass

def Create_model(args):
    # 固定使用内聚版 EM（encoder=gat_gt_serial, fusion=gt_fusion, MoCo 多视图）
    model = EM(
        feature=args.dimensions,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        decoder1=args.decoder1,
        dropout=args.dropout
    )
    # 模型参数统一初始化：仅对包含 weight 属性的模块应用 reset_parameters
    try:
        model.apply(_init_module)
    except Exception:
        # 初始化失败时不影响后续流程
        pass
    # 仅优化需要训练的参数（显式排除 requires_grad=False，例如 MoCo 动量分支/缓冲）
    trainable_params = (p for p in model.parameters() if getattr(p, "requires_grad", True))
    optimizer = Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer