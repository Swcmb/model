from typing import List, Optional, Any, Callable, Tuple, Union
from contextlib import nullcontext
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv
from parms_setting import settings
from utils import (
    em_path,
    maybe_rand_init_like,
    clamp_features,
    step_update,
    project_to_ball,
    # 轻量增强与读出等工具（从 utils 暴露到 layer 命名空间）
    reset_parameters,
    AvgReadout,
    random_permute_features,
    add_noise,
    attribute_mask,
    noise_then_mask,
    apply_augmentation,
)

# 全局参数
args = settings()

# =================================================
# 编码器：gat_gt_serial（底层组件）
# =================================================
class GATGTSerial(nn.Module):
    """
    先 GATConv 再 TransformerConv 的串联编码器
    
    该类实现了一个两层的图神经网络编码器，首先使用GATConv层进行节点特征提取，
    然后通过TransformerConv层进一步编码，实现图结构数据的表示学习。
    """
    def __init__(self, in_dim: int, hidden1: int, hidden2: int, dropout: float, gat_heads: int = 4):
        """
        初始化GATGTSerial编码器
        
        Args:
            in_dim (int): 输入特征维度
            hidden1 (int): 第一层GATConv的隐藏层维度
            hidden2 (int): 第二层TransformerConv的输出维度
            dropout (float): Dropout概率
            gat_heads (int, optional): GATConv层的注意力头数，默认为4
        """
        super().__init__()
        self.gat1 = GATConv(in_channels=in_dim, out_channels=hidden1, heads=gat_heads, concat=True, dropout=dropout)
        self.prelu_g1 = nn.PReLU(hidden1 * gat_heads)
        self.gt2 = TransformerConv(in_channels=hidden1 * gat_heads, out_channels=hidden2, heads=1, concat=False, dropout=dropout)
        self.prelu_t2 = nn.PReLU(hidden2)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        对图节点进行编码
        
        节点级编码：GAT -> Dropout -> Transformer -> PReLU
        
        Args:
            x (torch.Tensor): 节点特征矩阵，形状为 [num_nodes, in_dim]
            edge_index (torch.Tensor): 图的边索引矩阵，形状为 [2, num_edges]
            
        Returns:
            torch.Tensor: 编码后的节点表示，形状为 [num_nodes, hidden2]
        """
        x1 = self.prelu_g1(self.gat1(x, edge_index))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.gt2(x1, edge_index)
        x2 = self.prelu_t2(x2)
        return x2

# =================================================
# 融合策略抽象基类
# =================================================
class FusionStrategy(nn.Module, ABC):
    """
    融合策略抽象基类，支持热插拔
    
    定义了融合策略的接口，所有具体的融合策略都需要继承此类并实现forward方法
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        抽象前向传播方法，由子类具体实现
        
        Returns:
            torch.Tensor: 融合后的特征表示
        """
        pass

# =================================================
# 协作注意力组件
# =================================================

class PairwiseCoAttention(nn.Module):
    """
    成对协同注意力模块
    
    实现两个输入张量A和B之间的双向注意力机制，其中A关注B，B也关注A。
    每个方向都使用查询、键和值的线性变换，并应用注意力权重来聚合信息。
    
    Args:
        dim (int): 输入特征维度
        hidden_dim (Optional[int]): 隐藏层维度，如果为None则使用输入维度
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        """
        初始化成对协同注意力模块
        
        Args:
            dim (int): 输入特征维度
            hidden_dim (Optional[int]): 隐藏层维度，如果为None则使用输入维度
        """
        super().__init__()
        h = hidden_dim or dim

        # A -> B 方向的注意力计算
        # Wq_A: A的查询投影矩阵
        # Wk_B: B的键投影矩阵
        # Wv_B: B的值投影矩阵
        self.Wq_A = nn.Linear(dim, h)
        self.Wk_B = nn.Linear(dim, h)
        self.Wv_B = nn.Linear(dim, h)

        # B -> A 方向的注意力计算
        # Wq_B: B的查询投影矩阵
        # Wk_A: A的键投影矩阵
        # Wv_A: A的值投影矩阵
        self.Wq_B = nn.Linear(dim, h)
        self.Wk_A = nn.Linear(dim, h)
        self.Wv_A = nn.Linear(dim, h)

        # 输出投影层，将注意力输出映射回原始维度
        self.proj_A = nn.Linear(h, dim)
        self.proj_B = nn.Linear(h, dim)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，实现A和B之间的双向注意力
        
        Args:
            A (torch.Tensor): 第一个输入张量，形状为 [..., dim]
            B (torch.Tensor): 第二个输入张量，形状为 [..., dim]
            

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 更新后的A和B张量，形状与输入相同
        """
        # A关注B的注意力计算
        # 分别计算A的查询向量和B的键、值向量
        Q_A, K_B, V_B = self.Wq_A(A), self.Wk_B(B), self.Wv_B(B)
        # 计算注意力权重，使用缩放点积注意力公式
        attn_AB = torch.softmax(Q_A @ K_B.T / (K_B.size(1) ** 0.5), dim=1)
        # 使用注意力权重聚合B的值向量，并通过投影层得到更新部分
        A_out = A + self.proj_A(attn_AB @ V_B)

        # B关注A的注意力计算
        # 分别计算B的查询向量和A的键、值向量
        Q_B, K_A, V_A = self.Wq_B(B), self.Wk_A(A), self.Wv_A(A)
        # 计算注意力权重，使用缩放点积注意力公式
        attn_BA = torch.softmax(Q_B @ K_A.T / (K_A.size(1) ** 0.5), dim=1)
        # 使用注意力权重聚合A的值向量，并通过投影层得到更新部分
        B_out = B + self.proj_B(attn_BA @ V_A)

        return A_out, B_out

class TransformerPairwiseCoAttention(nn.Module):
    """
    Transformer双向协作注意力模块
    
    实现了两个输入序列之间的双向注意力机制，其中每个序列都可以作为查询去关注另一个序列。
    这种机制允许两个输入序列A和B之间进行细粒度的信息交互，并通过多头注意力和前馈网络
    进行特征增强。
    
    Attributes:
        dim (int): 输入特征的维度
        h (int): 注意力头的数量
        d (int): 每个注意力头的维度 (dim // num_heads)
        dropout (float): Dropout概率
        
        qA (nn.Linear): 序列A的查询投影矩阵
        kB (nn.Linear): 序列B的键投影矩阵
        vB (nn.Linear): 序列B的值投影矩阵
        qB (nn.Linear): 序列B的查询投影矩阵
        kA (nn.Linear): 序列A的键投影矩阵
        vA (nn.Linear): 序列A的值投影矩阵
        
        outA (nn.Linear): 序列A方向注意力输出的投影层
        outB (nn.Linear): 序列B方向注意力输出的投影层
        
        lnA1 (nn.LayerNorm): 序列A方向第一次层归一化
        lnA2 (nn.LayerNorm): 序列A方向第二次层归一化
        lnB1 (nn.LayerNorm): 序列B方向第一次层归一化
        lnB2 (nn.LayerNorm): 序列B方向第二次层归一化
        
        ffnA (nn.Sequential): 序列A方向的前馈网络
        ffnB (nn.Sequential): 序列B方向的前馈网络
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_gelu: bool = True,
    ):
        # 调用父类初始化方法
        super().__init__()
        # 确保维度可以被注意力头数整除，否则无法正确分割多头注意力
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        # 保存维度参数：总维度、头数、每个头的维度
        self.dim = dim          # 总特征维度
        self.h = num_heads      # 注意力头数
        self.d = dim // num_heads  # 每个注意力头的维度
        self.dropout = dropout  # Dropout概率

        # --- A -> B 方向的注意力投影矩阵 (查询来自A，键/值来自B) ---
        self.qA = nn.Linear(dim, dim)  # A的查询投影
        self.kB = nn.Linear(dim, dim)  # B的键投影
        self.vB = nn.Linear(dim, dim)  # B的值投影
        # --- B -> A 方向的注意力投影矩阵 (查询来自B，键/值来自A) ---
        self.qB = nn.Linear(dim, dim)  # B的查询投影
        self.kA = nn.Linear(dim, dim)  # A的键投影
        self.vA = nn.Linear(dim, dim)  # A的值投影

        # 多头注意力输出的投影层，用于合并多头特征
        self.outA = nn.Linear(dim, dim)  # A方向注意力输出的最终投影
        self.outB = nn.Linear(dim, dim)  # B方向注意力输出的最终投影

        # LayerNorm和前馈网络(FFN)，采用Transformer的设计风格
        # 每个方向都有自己的层归一化和前馈网络
        self.lnA1 = nn.LayerNorm(dim)  # A方向第一次层归一化
        self.lnA2 = nn.LayerNorm(dim)  # A方向第二次层归一化（FFN后）
        self.lnB1 = nn.LayerNorm(dim)  # B方向第一次层归一化
        self.lnB2 = nn.LayerNorm(dim)  # B方向第二次层归一化（FFN后）

        # 设置前馈网络隐藏层维度，如果未指定则使用4倍输入维度（Transformer的默认做法）
        ffn_hidden_dim = ffn_hidden_dim or (4 * dim)
        # 根据参数选择激活函数：GELU或ReLU
        act = nn.GELU() if use_gelu else nn.ReLU()
        # 构建A方向的前馈网络，包含两个线性层、激活函数和Dropout
        self.ffnA = nn.Sequential(
            nn.Linear(dim, ffn_hidden_dim),  # 升维
            act,                             # 非线性激活
            nn.Dropout(dropout),             # 随机失活
            nn.Linear(ffn_hidden_dim, dim),  # 降维回原维度
            nn.Dropout(dropout),             # 随机失活
        )
        # 构建B方向的前馈网络，结构与A方向相同但参数独立
        # share same FFN structure for B (separate params)
        self.ffnB = nn.Sequential(
            nn.Linear(dim, ffn_hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入张量分割成多个注意力头
        
        Args:
            x (torch.Tensor): 输入张量，形状为[B, dim]
            
        Returns:
            torch.Tensor: 分割后的张量，形状为[B, h, d]
        """
        B, D = x.shape
        return x.view(B, self.h, self.d)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        合并多个注意力头的结果
        
        Args:
            x (torch.Tensor): 输入张量，形状为[B, h, d]
            
        Returns:
            torch.Tensor: 合并后的张量，形状为[B, dim]
        """
        B = x.size(0)
        return x.contiguous().view(B, self.dim)

    def _pairwise_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        执行配对注意力计算
        
        Args:
            Q (torch.Tensor): 查询张量，形状为[B, h, d]
            K (torch.Tensor): 键张量，形状为[B, h, d]
            V (torch.Tensor): 值张量，形状为[B, h, d]
            
        Returns:
            torch.Tensor: 注意力计算结果，形状为[B, dim]
        """
        # scaled dot product per head -> [B, h]
        scores = (Q * K).sum(dim=-1) / (self.d ** 0.5)  # [B, h]
        # softmax across heads (dim=-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, h, 1]
        # weight V heads and merge
        weighted = weights * V  # [B, h, d]
        merged = self._merge_heads(weighted)  # [B, dim]
        return merged

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数
        
        Args:
            A (torch.Tensor): 第一个输入序列，形状为[B, dim]
            B (torch.Tensor): 第二个输入序列，形状为[B, dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 处理后的A和B序列，形状均为[B, dim]
        """
        # --- compute QKV per direction and split into heads ---
        Q_A = self._split_heads(self.qA(A))  # [B, h, d]
        K_B = self._split_heads(self.kB(B))
        V_B = self._split_heads(self.vB(B))

        Q_B = self._split_heads(self.qB(B))
        K_A = self._split_heads(self.kA(A))
        V_A = self._split_heads(self.vA(A))

        # --- A attends to B ---
        attn_out_A = self._pairwise_attention(Q_A, K_B, V_B)  # [B, dim]
        attn_out_A = self.outA(attn_out_A)  # projection
        attn_out_A = F.dropout(attn_out_A, p=self.dropout, training=self.training)

        # residual + norm
        A_res = self.lnA1(A + attn_out_A)
        # FFN with residual
        A_ffn = self.ffnA(A_res)
        A_out = self.lnA2(A_res + A_ffn)

        # --- B attends to A ---
        attn_out_B = self._pairwise_attention(Q_B, K_A, V_A)
        attn_out_B = self.outB(attn_out_B)
        attn_out_B = F.dropout(attn_out_B, p=self.dropout, training=self.training)

        B_res = self.lnB1(B + attn_out_B)
        B_ffn = self.ffnB(B_res)
        B_out = self.lnB2(B_res + B_ffn)

        return A_out, B_out

class MultiHeadPairwiseCoAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # A -> B direction
        self.qA = nn.Linear(dim, dim)
        self.kB = nn.Linear(dim, dim)
        self.vB = nn.Linear(dim, dim)

        # B -> A direction
        self.qB = nn.Linear(dim, dim)
        self.kA = nn.Linear(dim, dim)
        self.vA = nn.Linear(dim, dim)

        self.outA = nn.Linear(dim, dim)
        self.outB = nn.Linear(dim, dim)

        self.scale = self.head_dim ** -0.5

    def _split(self, x):
        B, L, D = x.size()
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x):
        B, H, L, D = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, H * D)

    def forward(self, A, B):
        qA = self._split(self.qA(A))
        kB = self._split(self.kB(B))
        vB = self._split(self.vB(B))

        qB = self._split(self.qB(B))
        kA = self._split(self.kA(A))
        vA = self._split(self.vA(A))

        attn_AB = torch.softmax((qA @ kB.transpose(-2, -1)) * self.scale, dim=-1)
        attn_BA = torch.softmax((qB @ kA.transpose(-2, -1)) * self.scale, dim=-1)

        A2B = attn_AB @ vB
        B2A = attn_BA @ vA

        A2B = self.outA(self._merge(A2B))
        B2A = self.outB(self._merge(B2A))

        return A2B, B2A

class MultiHeadCoAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gA = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.gB = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.out = nn.Linear(dim * 2, dim)

    def forward(self, A2B, B2A):
        gA = self.gA(torch.cat([A2B, B2A], dim=-1))
        gB = self.gB(torch.cat([A2B, B2A], dim=-1))
        fused = torch.cat([
            gA * A2B + (1 - gA) * B2A,
            gB * B2A + (1 - gB) * A2B
        ], dim=-1)
        return self.out(fused)

class MultiHeadTransformerPairwiseCoAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.self_attn_A = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_B = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.co_attn = MultiHeadPairwiseCoAttention(dim, num_heads=num_heads)
        self.fusion = MultiHeadCoAttentionFusion(dim)

        self.norm1_A = nn.LayerNorm(dim)
        self.norm1_B = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, A, B):
        A2, _ = self.self_attn_A(A, A, A)
        B2, _ = self.self_attn_B(B, B, B)
        A = self.norm1_A(A + A2)
        B = self.norm1_B(B + B2)

        A2B, B2A = self.co_attn(A, B)
        Z = self.fusion(A2B, B2A)

        Z = self.norm2(Z + self.mlp(Z))
        return Z





# =================================================
# 注意力机制工厂类 - 支持命令行配置
# =================================================

class AttentionFactory:
    """
    注意力机制工厂类，支持通过配置字符串动态创建不同类型的注意力模块
    """
    
    @staticmethod
    def create_co_attention(dim: int, config_str: str = "transformer", **kwargs) -> nn.Module:
        """
        根据配置字符串创建协作注意力模块
        
        Args:
            dim (int): 输入特征维度
            config_str (str): 配置字符串，格式为 "type[param1=value1,param2=value2,...]"
            **kwargs: 额外参数
            
        Returns:
            nn.Module: 协作注意力模块
        """
        # 解析配置字符串
        if '[' in config_str and ']' in config_str:
            attention_type = config_str.split('[')[0]
            params_str = config_str.split('[')[1].split(']')[0]
            params = AttentionFactory._parse_params(params_str)
        else:
            attention_type = config_str
            params = {}
        
        # 合并参数
        params.update(kwargs)
        
        # 根据类型创建注意力模块
        if attention_type == "pairwise":
            return PairwiseCoAttention(dim, params.get('hidden_dim', dim))
        
        elif attention_type == "transformer":
            return TransformerPairwiseCoAttention(
                dim=dim,
                num_heads=params.get('num_heads', 4),
                ffn_hidden_dim=params.get('ffn_hidden_dim', 4 * dim),
                dropout=params.get('dropout', 0.0),
                use_gelu=params.get('use_gelu', True)
            )
        
        elif attention_type == "multihead":
            return MultiHeadPairwiseCoAttention(
                dim=dim,
                num_heads=params.get('num_heads', 4)
            )
        
        
        
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
    
    @staticmethod
    def create_fusion_attention(dim: int, config_str: str = "self_attention", **kwargs) -> nn.Module:
        """
        根据配置字符串创建融合注意力模块
        
        Args:
            dim (int): 输入特征维度
            config_str (str): 配置字符串
            **kwargs: 额外参数
            
        Returns:
            nn.Module: 融合注意力模块
        """
        if '[' in config_str and ']' in config_str:
            fusion_type = config_str.split('[')[0]
            params_str = config_str.split('[')[1].split(']')[0]
            params = AttentionFactory._parse_params(params_str)
        else:
            fusion_type = config_str
            params = {}
        
        params.update(kwargs)
        
        if fusion_type == "self_attention":
            return SelfAttentionFusion(
                hidden_dim=dim,
                heads=params.get('heads', 4),
                dropout=params.get('dropout', 0.1)
            )
        
        elif fusion_type == "co_attention":
            co_attention_config = params.get('co_attention_type', 'transformer')
            return CoAttentionFusion(
                hidden_dim=dim,
                hidden_dim_co=params.get('hidden_dim_co', dim),
                num_heads=params.get('num_heads', 4),
                dropout=params.get('dropout', 0.1),
                co_attention_type=co_attention_config
            )
        
        elif fusion_type == "hybrid":
            co_attention_config = params.get('co_attention_type', 'transformer')
            return HybridFusion(
                hidden_dim=dim,
                heads=params.get('heads', 4),
                dropout=params.get('dropout', 0.1),
                co_hidden_dim=params.get('co_hidden_dim', dim),
                fusion_weight=params.get('fusion_weight', 0.5),
                co_attention_type=co_attention_config
            )
        
        elif fusion_type == "transformer_multihead":
            return MultiHeadTransformerPairwiseCoAttention(
                dim=dim,
                num_heads=params.get('num_heads', 4),
                dropout=params.get('dropout', 0.1)
            )
        
        
        
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    @staticmethod
    def _parse_params(params_str: str) -> dict:
        """解析参数字符串为字典"""
        params = {}
        if not params_str:
            return params
        
        for param in params_str.split(','):
            if '=' in param:
                key, value = param.split('=', 1)
                # 尝试转换为适当的类型
                if value.lower() in ['true', 'false']:
                    params[key.strip()] = value.lower() == 'true'
                elif value.replace('.', '', 1).isdigit():
                    params[key.strip()] = float(value) if '.' in value else int(value)
                else:
                    params[key.strip()] = value.strip()
        
        return params
    
    @staticmethod
    def get_available_configurations() -> dict:
        """获取所有可用的配置选项"""
        return {
            "co_attention_types": [
                "pairwise",
                "transformer[num_heads=4,dropout=0.1]",
                "multihead[num_heads=4]", 
                
            ],
            "fusion_types": [
                "self_attention[heads=4,dropout=0.1]",
                "co_attention[co_attention_type=transformer]",
                "hybrid[fusion_weight=0.5,co_attention_type=transformer]",
                "transformer_multihead[num_heads=4]",
                
            ]
        }

# =================================================
# 具体融合策略实现
# =================================================

class SelfAttentionFusion(FusionStrategy):
    """
    原始的自注意力融合策略
    
    使用多头自注意力机制对两个实体进行融合，通过堆叠两个实体表示，
    然后应用自注意力和前馈网络进行特征融合。
    
    Attributes:
        hidden_dim (int): 隐藏层维度
        heads (int): 注意力头数
        dropout (float): Dropout概率
        mha (nn.MultiheadAttention): 多头注意力层
        ffn (nn.Sequential): 前馈网络
        norm1 (nn.LayerNorm): 第一层归一化
        norm2 (nn.LayerNorm): 第二层归一化
        dropout_layer (nn.Dropout): Dropout层
    """
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1):
        """
        初始化自注意力融合策略
        
        Args:
            hidden_dim (int): 隐藏层维度
            heads (int): 注意力头数，默认为4
            dropout (float): Dropout概率，默认为0.1
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        
        # 直接初始化模块
        self.mha = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, 
            num_heads=self.heads, 
            dropout=self.dropout, 
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，使用自注意力机制融合两个实体
        
        Args:
            e1 (torch.Tensor): 第一个实体的特征表示，形状为[batch_size, hidden_dim]
            e2 (torch.Tensor): 第二个实体的特征表示，形状为[batch_size, hidden_dim]
            
        Returns:
            torch.Tensor: 融合后的特征表示，形状为[batch_size, 2*hidden_dim]
        """
        B, H = e1.size(0), e1.size(1)
        x = torch.stack([e1, e2], dim=1)          # [B,2,H]
        attn_out, _ = self.mha(x, x, x)           # [B,2,H]
        x = self.norm1(x + self.dropout_layer(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout_layer(ffn_out)) # [B,2,H]
        x = x.reshape(B, 2 * H)                   # [B,2H]
        return x

class CoAttentionFusion2(FusionStrategy):
    """
    协作注意力融合策略，使用PairwiseCoAttention
    
    使用协作注意力机制对两个实体进行融合，通过PairwiseCoAttention模块
    实现两个实体之间的相互关注，然后拼接结果。
    
    Attributes:
        hidden_dim (int): 隐藏层维度
        hidden_dim_co (int): 协作注意力的隐藏层维度
        co_attn (PairwiseCoAttention): 协作注意力模块
    """
    def __init__(self, hidden_dim: int, hidden_dim_co: Optional[int] = None):
        """
        初始化协作注意力融合策略
        
        Args:
            hidden_dim (int): 隐藏层维度
            hidden_dim_co (Optional[int]): 协作注意力的隐藏层维度，默认为None
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_dim_co = hidden_dim_co = hidden_dim
        self.co_attn = PairwiseCoAttention(self.hidden_dim, self.hidden_dim_co)
    
    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，使用协作注意力机制融合两个实体
        
        Args:
            e1 (torch.Tensor): 第一个实体的特征表示，形状为[batch_size, hidden_dim]
            e2 (torch.Tensor): 第二个实体的特征表示，形状为[batch_size, hidden_dim]
            
        Returns:
            torch.Tensor: 融合后的特征表示，形状为[batch_size, 2*hidden_dim]
        """
        e1_out, e2_out = self.co_attn(e1, e2)
        # 拼接两个实体的表示
        return torch.cat([e1_out, e2_out], dim=1)  # [B, 2H]

class CoAttentionFusion(FusionStrategy):
    """
    协作注意力融合策略，可以选择使用TransformerPairwiseCoAttention或PairwiseCoAttention
    
    使用协作注意力机制对两个实体进行融合，根据参数选择使用TransformerPairwiseCoAttention模块
    或PairwiseCoAttention模块实现两个实体之间的相互关注，然后拼接结果。
    
    Attributes:
        hidden_dim (int): 隐藏层维度
        hidden_dim_co (int): 协作注意力的隐藏层维度
        co_attn (Union[PairwiseCoAttention, TransformerPairwiseCoAttention]): 协作注意力模块
    """
    def __init__(
        self,
        hidden_dim: int,
        hidden_dim_co: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        co_attention_type: str = 'transformer'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_dim_co = hidden_dim if hidden_dim_co is None else hidden_dim_co

        # 根据参数选择使用哪种类型的协作注意力
        if co_attention_type == 'pairwise':
            self.co_attn = PairwiseCoAttention(
                dim=self.hidden_dim,
                hidden_dim=self.hidden_dim_co
            )
        else:  # 默认使用 transformer 类型
            self.co_attn = TransformerPairwiseCoAttention(
                dim=self.hidden_dim,
                num_heads=num_heads,
                ffn_hidden_dim=4 * self.hidden_dim,  # Transformer 默认
                dropout=dropout,
            )

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """
        输入:
            e1: [B, H]
            e2: [B, H]
        输出:
            [B, 2H]
        """
        e1_out, e2_out = self.co_attn(e1, e2)
        return torch.cat([e1_out, e2_out], dim=1)

class HybridFusion(FusionStrategy):
    """
    混合融合策略：结合自注意力和协作注意力
    
    结合自注意力和协作注意力两种机制，通过加权融合的方式
    得到最终的融合结果。
    
    Attributes:
        hidden_dim (int): 隐藏层维度
        heads (int): 注意力头数
        dropout (float): Dropout概率
        co_hidden_dim (Optional[int]): 协作注意力的隐藏层维度
        fusion_weight (float): 自注意力融合结果的权重
        self_attn_fusion (SelfAttentionFusion): 自注意力融合模块
        co_attn_fusion (CoAttentionFusion): 协作注意力融合模块
    """
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1, 
                 co_hidden_dim: Optional[int] = None, fusion_weight: float = 0.5,
                 co_attention_type: str = 'transformer'):
        """
        初始化混合融合策略
        
        Args:
            hidden_dim (int): 隐藏层维度
            heads (int): 注意力头数，默认为4
            dropout (float): Dropout概率，默认为0.1
            co_hidden_dim (Optional[int]): 协作注意力的隐藏层维度，默认为None
            fusion_weight (float): 自注意力融合结果的权重，默认为0.5
            co_attention_type (str): 协作注意力类型，默认为 'transformer'
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.co_hidden_dim = co_hidden_dim
        self.fusion_weight = fusion_weight
        
        self.self_attn_fusion = SelfAttentionFusion(self.hidden_dim, self.heads, self.dropout)
        self.co_attn_fusion = CoAttentionFusion(self.hidden_dim, hidden_dim_co=co_hidden_dim, 
                                               co_attention_type=co_attention_type)
    
    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，混合自注意力和协作注意力机制融合两个实体
        
        Args:
            e1 (torch.Tensor): 第一个实体的特征表示，形状为[batch_size, hidden_dim]
            e2 (torch.Tensor): 第二个实体的特征表示，形状为[batch_size, hidden_dim]
            
        Returns:
            torch.Tensor: 融合后的特征表示，形状为[batch_size, 2*hidden_dim]
        """
        # 自注意力融合
        self_attn_out = self.self_attn_fusion.forward(e1, e2)
        # 协作注意力融合
        co_attn_out = self.co_attn_fusion.forward(e1, e2)
        
        # 加权融合
        return self.fusion_weight * self_attn_out + (1 - self.fusion_weight) * co_attn_out

# =================================================
# Graph Transformer 风格融合策略
# =================================================

class GraphTransformerStyleFusion(nn.Module):
    """
    Graph Transformer 风格的两-token注意力 + 前馈；输出拼接为 [B, 2H]
    增强版：支持协作注意力和热插拔功能
    
    该模块将两个实体表示作为输入，通过可配置的融合策略进行特征融合，
    最终将两个实体的表示拼接为一个长度为2H的向量。支持多种融合策略的
    动态切换，实现热插拔功能。
    
    Attributes:
        _strategy (FusionStrategy): 融合策略实例，支持热插拔
        hidden_dim (int): 隐藏层维度
        heads (int): 注意力头的数量
        dropout (float): Dropout概率
        strategy_type (str): 当前使用的策略类型
        strategy_kwargs (dict): 策略特定的额外参数
        _preserve_legacy (bool): 是否保持向后兼容的原始模块
        mha (nn.MultiheadAttention): 多头注意力层（仅在self_attention模式下使用）
        ffn (nn.Sequential): 前馈网络（仅在self_attention模式下使用）
        norm1 (nn.LayerNorm): 第一层归一化（仅在self_attention模式下使用）
        norm2 (nn.LayerNorm): 第二层归一化（仅在self_attention模式下使用）
        _dropout (nn.Dropout): Dropout层（仅在self_attention模式下使用）
    """
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1, 
                 strategy_type: str = "self_attention", **strategy_kwargs):
        """
        初始化GraphTransformerStyleFusion模块
        
        Args:
            hidden_dim (int): 隐藏层维度
            heads (int, optional): 注意力头的数量，默认为4
            dropout (float, optional): Dropout概率，默认为0.1
            strategy_type (str, optional): 融合策略类型，可选 "self_attention", "co_attention", "hybrid"
            **strategy_kwargs: 策略特定的额外参数
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.strategy_type = strategy_type
        self.strategy_kwargs = strategy_kwargs
        
        # 初始化融合策略
        self._strategy = None
        self._initialize_strategy()
        
        # 保持向后兼容的原始模块（用于self_attention模式）
        self._preserve_legacy = strategy_type == "self_attention"
        if self._preserve_legacy:
            self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self._dropout = nn.Dropout(dropout)

    def _initialize_strategy(self):
        """初始化融合策略"""
        try:
            # 首先尝试使用工厂方法创建融合策略
            self._strategy = AttentionFactory.create_fusion_attention(
                dim=self.hidden_dim,
                config_str=self.strategy_type,
                heads=self.heads,
                dropout=self.dropout,
                **self.strategy_kwargs
            )
        except Exception:
            # 如果工厂方法失败，回退到原有的策略创建方式
            if self.strategy_type == "self_attention":
                self._strategy = SelfAttentionFusion(self.hidden_dim, self.heads, self.dropout)
            elif self.strategy_type == "co_attention":
                # 处理参数名映射
                kwargs = self.strategy_kwargs.copy()
                if 'co_hidden_dim' in kwargs:
                    kwargs['hidden_dim_co'] = kwargs.pop('co_hidden_dim')
                # 添加协作注意力类型参数
                kwargs['co_attention_type'] = getattr(args, 'co_attention_type', 'transformer')
                self._strategy = CoAttentionFusion(self.hidden_dim, **kwargs)
            elif self.strategy_type == "hybrid":
                # 处理参数名映射
                kwargs = self.strategy_kwargs.copy()
                if 'co_hidden_dim' in kwargs:
                    kwargs['co_hidden_dim'] = kwargs.pop('co_hidden_dim')
                # 添加协作注意力类型参数
                kwargs['co_attention_type'] = getattr(args, 'co_attention_type', 'transformer')
                self._strategy = HybridFusion(self.hidden_dim, self.heads, self.dropout, **kwargs)
            else:
                raise ValueError(f"Unknown strategy_type: {self.strategy_type}")

    def set_strategy(self, strategy_type: str, **strategy_kwargs):
        """
        热插拔：动态设置融合策略
        
        Args:
            strategy_type (str): 新的融合策略类型
            **strategy_kwargs: 策略特定的额外参数
        """
        self.strategy_type = strategy_type
        self.strategy_kwargs = strategy_kwargs
            
        # 初始化新策略
        self._initialize_strategy()
        self._preserve_legacy = (strategy_type == "self_attention")

    def get_available_strategies(self) -> List[str]:
        """获取可用的融合策略列表
        
        Returns:
            List[str]: 可用的融合策略列表
        """
        return ["self_attention", "co_attention", "hybrid"]

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，对两个实体表示进行特征融合
        
        将两个实体表示通过配置的策略进行融合
        
        Args:
            e1 (torch.Tensor): 第一个实体的表示，形状为 [B, H]
            e2 (torch.Tensor): 第二个实体的表示，形状为 [B, H]
            
        Returns:
            torch.Tensor: 融合后的表示，形状为 [B, 2H]
        """
        if self._preserve_legacy and hasattr(self, 'mha'):
            # 保持原始实现的兼容性
            B, H = e1.size(0), e1.size(1)
            x = torch.stack([e1, e2], dim=1)          # [B,2,H]
            attn_out, _ = self.mha(x, x, x)           # [B,2,H]
            x = self.norm1(x + self._dropout(attn_out))
            ffn_out = self.ffn(x)
            x = self.norm2(x + self._dropout(ffn_out)) # [B,2,H]
            x = x.reshape(B, 2 * H)                   # [B,2H]
            return x
        else:
            # 使用新策略
            return self._strategy.forward(e1, e2)

# =================================================
# 融合解码器
# =================================================

class FusionDecoder(nn.Module):
    """
    融合解码器，用于将两个实体的表示融合后解码为二分类分数
    增强版：支持多种注意力机制的工厂模式配置
    
    该解码器使用工厂模式创建的融合策略将两个实体表示融合，
    然后通过两层全连接网络将融合后的特征映射到最终的二分类分数。
    
    Attributes:
        hidden_dim (int): 隐藏层维度
        fusion (Union[FusionStrategy, nn.Module]): 融合策略模块
        strategy (Union[FusionStrategy, nn.Module]): fusion属性的别名，用于向后兼容
        proj4h (nn.Linear): 将2H维度映射到4H维度的线性变换层
        fc1 (nn.Linear): 第一层全连接层
        fc2 (nn.Linear): 第二层全连接层，输出维度为1
    """
    def __init__(self, hidden_dim: int, decoder1_dim: int, heads: int = 4, dropout: float = 0.1,
                 fusion_strategy: str = "self_attention", **fusion_kwargs):
        """
        初始化融合解码器
        
        Args:
            hidden_dim (int): 隐藏层维度
            decoder1_dim (int): 第一层解码器的输出维度
            heads (int, optional): 注意力头数，默认为4
            dropout (float, optional): Dropout概率，默认为0.1
            fusion_strategy (str, optional): 融合策略，支持工厂模式配置
            **fusion_kwargs: 融合策略的额外参数
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 使用工厂模式创建融合模块
        try:
            self.fusion = AttentionFactory.create_fusion_attention(
                dim=hidden_dim,
                config_str=fusion_strategy,
                heads=heads,
                dropout=dropout,
                **fusion_kwargs
            )
        except Exception as e:
            print(f"Warning: Failed to create fusion with factory method: {e}")
            print("Falling back to GraphTransformerStyleFusion...")
            # 回退到原有方法
            self.fusion = GraphTransformerStyleFusion(
                hidden_dim=hidden_dim, 
                heads=heads, 
                dropout=dropout, 
                strategy_type=fusion_strategy, 
                **fusion_kwargs
            )
        
        # 保持向后兼容性的别名
        self.strategy = self.fusion
        
        self.proj4h: nn.Module = nn.Linear(2 * hidden_dim, 4 * hidden_dim)  # 将[2H]映射到[4H]
        self.fc1 = nn.Linear(4 * hidden_dim, decoder1_dim)
        self.fc2 = nn.Linear(decoder1_dim, 1)

    def set_fusion_strategy(self, strategy_type: str, **strategy_kwargs):
        """
        热插拔：动态设置融合策略
        
        Args:
            strategy_type (str): 新的融合策略类型
            **strategy_kwargs: 策略特定的额外参数
        """
        self.fusion.set_strategy(strategy_type, **strategy_kwargs)
        # 保持别名同步
        self.strategy = self.fusion

    def get_fusion_info(self) -> dict:
        """
        获取当前融合策略信息
        
        Returns:
            dict: 包含策略类型和相关参数的字典
        """
        return {
            "strategy_type": self.fusion.strategy_type,
            "strategy_kwargs": self.fusion.strategy_kwargs,
            "available_strategies": self.fusion.get_available_strategies()
        }

    def forward(self, e1: torch.Tensor, e2: torch.Tensor):
        """
        前向传播函数，将两个实体表示融合并解码为二分类分数
        
        Args:
            e1 (torch.Tensor): 第一个实体的表示，形状为 [B, H]
            e2 (torch.Tensor): 第二个实体的表示，形状为 [B, H]
            
        Returns:
            tuple: 包含以下元素的元组：
                - log (torch.Tensor): 解码后的二分类分数，形状为 [B, 1]
                - log1 (torch.Tensor): 第一层全连接网络的输出，形状为 [B, decoder1_dim]
        """
        # 两实体表示先融合再解码，串联两层全连接输出二分类分数
        feat2h = self.fusion(e1, e2)              # [B,2H]
        fused4h = self.proj4h(feat2h)             # [B,4H]
        log1 = F.relu(self.fc1(fused4h))          # [B,decoder1]
        log = self.fc2(log1)                      # [B,1]
        return log, log1

# =================================================
# MoCo 多视图工厂类 - 支持命令行配置
# =================================================

class ModelFactory:
    """
    模型工厂类，支持通过配置字符串动态创建不同类型的自监督学习模块
    
    支持的模型类型：
    - moco: MoCo多视图对比学习模型
    - byol: BYOL多视图自监督学习模型
    """
    
    @staticmethod
    def create_model(model_type: str, base_dim: int, proj_dim: int, num_views: int, 
                   config_str: str = "basic", **kwargs) -> nn.Module:
        """
        根据模型类型和配置字符串创建自监督学习模块
        
        Args:
            model_type (str): 模型类型 ("moco" 或 "byol")
            base_dim (int): 基础维度，即输入特征的维度
            proj_dim (int): 投影维度，即投影头输出特征的维度
            num_views (int): 视图数量
            config_str (str): 配置字符串，格式为 "type[param1=value1,...]"
            **kwargs: 额外参数
            
        Returns:
            nn.Module: 自监督学习模块
        """
        # 解析配置字符串
        if '[' in config_str and ']' in config_str:
            config_type = config_str.split('[')[0]
            params_str = config_str.split('[')[1].split(']')[0]
            params = ModelFactory._parse_params(params_str)
        else:
            config_type = config_str
            params = {}
        
        # 合并参数
        params.update(kwargs)
        
        # 根据模型类型创建对应模块
        if model_type.lower() == "moco":
            return ModelFactory._create_moco_model(base_dim, proj_dim, num_views, config_type, params)
        elif model_type.lower() == "byol":
            return ModelFactory._create_byol_model(base_dim, proj_dim, num_views, config_type, params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def _create_moco_model(base_dim: int, proj_dim: int, num_views: int, 
                          config_type: str, params: dict) -> nn.Module:
        """创建MoCo模型"""
        if config_type == "basic":
            return MoCoV2MultiView(
                base_dim=base_dim,
                proj_dim=proj_dim,
                num_views=num_views,
                K=params.get('K', 4096),
                m=params.get('m', 0.999),
                T=params.get('T', 0.2),
                queue_warmup_steps=params.get('queue_warmup_steps', 0),
                debug=params.get('debug', False)
            )
        elif config_type == "double_tau":
            return MoCoV2MultiViewDoubleTau(
                base_dim=base_dim,
                proj_dim=proj_dim,
                num_views=num_views,
                K=params.get('K', 4096),
                m=params.get('m', 0.999),
                tau1=params.get('tau1', 0.2),
                tau2=params.get('tau2', None),
                queue_warmup_steps=params.get('queue_warmup_steps', 0),
                debug=params.get('debug', False)
            )
        else:
            raise ValueError(f"Unknown MoCo type: {config_type}")
    
    @staticmethod
    def _create_byol_model(base_dim: int, proj_dim: int, num_views: int, 
                          config_type: str, params: dict) -> nn.Module:
        """创建BYOL模型"""
        if config_type == "basic":
            return BYOLMultiView(
                base_dim=base_dim,
                proj_dim=proj_dim,
                num_views=num_views,
                predictor_dim=params.get('predictor_dim', 256),
                m=params.get('m', 0.996),
                temperature=params.get('temperature', 0.2),
                debug=params.get('debug', False)
            )
        else:
            raise ValueError(f"Unknown BYOL type: {config_type}")
    
    @staticmethod
    def _parse_params(params_str: str) -> dict:
        """解析参数字符串为字典"""
        params = {}
        if not params_str:
            return params
        
        for param in params_str.split(','):
            if '=' in param:
                key, value = param.split('=', 1)
                # 尝试转换为适当的类型
                if value.lower() in ['true', 'false']:
                    params[key.strip()] = value.lower() == 'true'
                elif value.replace('.', '', 1).isdigit():
                    params[key.strip()] = float(value) if '.' in value else int(value)
                else:
                    params[key.strip()] = value.strip()
        
        return params
    
    @staticmethod
    def get_available_configurations() -> dict:
        """获取所有可用的模型配置选项"""
        return {
            "model_types": ["moco", "byol"],
            "moco_configs": [
                "basic[K=4096,m=0.999,T=0.2]",
                "double_tau[K=4096,m=0.999,tau1=0.2,tau2=0.3]"
            ],
            "byol_configs": [
                "basic[predictor_dim=256,m=0.996,temperature=0.2]"
            ],
            "descriptions": {
                "moco": "MoCo v2多视图对比学习模型，使用队列存储负样本",
                "byol": "BYOL多视图自监督学习模型，无需负样本的对称对比学习",
                "basic": "基础实现，简单高效"
            }
        }
    
    @staticmethod
    def get_comparison_table() -> str:
        """获取模型类型比较说明"""
        return """
        模型类型比较说明：
        
        | 特性 | MoCo | BYOL |
        |------|-------|-------|
        | 学习策略 | 对比学习（负样本） | 对称学习（无负样本） |
        | 队列管理 | 环形队列存储负样本 | 无队列 |
        | 网络结构 | 查询+键投影头 | 在线+目标网络+预测头 |
        | 更新机制 | 动量更新键编码器 | EMA更新目标网络 |
        | 损失函数 | InfoNCE对比损失 | 对称余弦相似度损失 |
        | 计算复杂度 | 中等（B×K） | 低（B×B） |
        | 适用场景 | 一般对比学习任务 | 大规模自监督学习 |
        | 参数调优 | 温度系数、队列大小 | EMA系数、预测头维度 |
        
        MoCo类型说明：
        
        | 特性 | Basic | DoubleTau |
        |------|--------|----------|
        | 队列管理 | 基础环形队列 | 基础环形队列 |
        | 温度控制 | 固定温度 | 双温度参数(τ₁,τ₂) |
        | 负样本筛选 | 全量 | 全量 |
        | 计算复杂度 | 低(B×K) | 低(B×K) |
        | 适用场景 | 一般对比学习 | 精细温度控制 |
        | 队列权重 | 无 | 无 |
        | 温度参数 | T=0.2 | tau1=0.2,tau2=0.3 |
        
        BYOL类型说明：
        
        | 特性 | Basic |
        |------|--------|
        | 预测头维度 | 256 |
        | EMA系数 | 0.996 |
        | 温度系数 | 0.2 |
        | 停止梯度 | 目标网络 |
        | 对称损失 | 温度缩放余弦相似度 |
        | 计算复杂度 | 低(B×B) |
        | 适用场景 | 大规模自监督学习 |
        | 更新策略 | 动量更新 |
        """

# =================================================
# MoCo 多视图实现
# =================================================
class MoCoV2MultiView(nn.Module):
    """
    多视图 MoCo v2实现
    
    该类实现了多视图版本的MoCo v2算法，主要特点包括：
    - 所有视图共享同一个查询(q)投影头
    - 每个视图拥有独立的键(k)投影头和队列
    - 返回每个视图的(logits, targets)元组用于对比学习
    
    Attributes:
        num_views (int): 视图数量
        K (int): 队列大小
        m (float): 动量更新系数
        T (float): 温度系数
        queue_warmup_steps (int): 队列预热步数
        debug (bool): 是否启用调试模式
        global_step (int): 全局训练步数
        q_proj (nn.Sequential): 查询投影头网络
        k_projs (nn.ModuleList): 键投影头网络列表
        
    Methods:
        momentum_update_key_encoders: 动量更新所有键编码器
        _dequeue_and_enqueue: 将键特征入队到对应视图的队列中
        forward: 前向传播，计算对比损失的logits和targets
    """
    def __init__(self, base_dim: int, proj_dim: int, num_views: int, K: int = 4096, m: float = 0.999, T: float = 0.2, queue_warmup_steps: int = 0, debug: bool = False):
        """
        初始化多视图MoCo v2模型
        
        Args:
            base_dim (int): 基础维度，即输入特征的维度
            proj_dim (int): 投影维度，即投影头输出特征的维度
            num_views (int): 视图数量，必须大于等于1
            K (int, optional): 队列大小，默认为4096
            m (float, optional): 动量更新系数，默认为0.999
            T (float, optional): 温度系数，默认为0.2
            queue_warmup_steps (int, optional): 队列预热步数，默认为0
            debug (bool, optional): 是否启用调试模式，默认为False
        """
        super().__init__()
        assert num_views >= 1, "num_views must be >= 1"
        assert proj_dim is not None and proj_dim > 0, "proj_dim must be positive"
        self.num_views = int(num_views)
        self.K = int(K)
        self.m = float(m)
        self.T = float(T)
        self.queue_warmup_steps = int(queue_warmup_steps)
        self.debug = bool(debug)
        self.global_step = 0
        self._filled = [0 for _ in range(self.num_views)]

        # ✅ 修复5：修正 warmup 判断为 < 而非 <=，确保 warmup_steps=0 时不跳过
        # warmup = self.global_step < self.queue_warmup_steps

        # 共享 q 投影头
        self.q_proj = nn.Sequential(
            nn.Linear(base_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        # 独立 k 投影头
        self.k_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_dim, proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim),
            ) for _ in range(self.num_views)
        ])
        # 初始化各 k_proj = q_proj，且冻结梯度
        with torch.no_grad():
            for k_proj in self.k_projs:
                for qp, kp in zip(self.q_proj.parameters(), k_proj.parameters()):
                    kp.data.copy_(qp.data)
                    kp.requires_grad = False

        # 为每个视图注册独立队列与指针
        for i in range(self.num_views):
            self.register_buffer(f"queue_{i}", F.normalize(torch.randn(proj_dim, self.K), dim=0))
            self.register_buffer(f"queue_ptr_{i}", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update_key_encoders(self):
        """
        使用动量更新策略更新所有键编码器的参数
        
        将每个键编码器的参数向对应的查询编码器参数靠近，
        更新公式为: param_k = param_k * m + param_q * (1-m)
        """
        # 动量更新：将 k 编码器向 q 编码器移动
        for k_proj in self.k_projs:
            for param_q, param_k in zip(self.q_proj.parameters(), k_proj.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor, view_idx: int):
        """
        将键特征入队到指定视图的队列中
        
        实现环形队列的出队入队操作，支持批量更新队列
        
        Args:
            keys (torch.Tensor): 待入队的键特征张量，形状为[B, C]
            view_idx (int): 视图索引，指定要更新的队列
        """
        # 出队入队：环形队列更新（安全切片，避免越界）
        keys = keys.detach()
        n_new = int(keys.shape[0])
        if n_new <= 0:
            return
        queue = getattr(self, f"queue_{view_idx}")
        queue_ptr = getattr(self, f"queue_ptr_{view_idx}")
        K = int(queue.size(1))  # 队列列数作为容量
        ptr = int(queue_ptr.item())
        # 统一用转置后的 [C, B] 视图进行列区间写入
        kT = keys.t()  # [C, n_new]

        # 剩余容量
        rem = K - ptr
        if n_new <= rem:
            # 单段写入：完全装入尾部
            queue[:, ptr:ptr + n_new] = kT[:, :n_new]
            ptr = (ptr + n_new) % K
        else:
            # 两段写入：尾段 + 头段
            len1 = rem
            if len1 > 0:
                queue[:, ptr:ptr + len1] = kT[:, :len1]
            len2 = min(n_new - len1, K)  # 头段长度不超过 K
            if len2 > 0:
                queue[:, 0:len2] = kT[:, len1:len1 + len2]
            ptr = (ptr + n_new) % K

        queue_ptr[0] = ptr

    def forward(self, q_embed: torch.Tensor, k_embeds: List[torch.Tensor]):
        """
        前向传播函数
        
        计算每个视图的对比学习logits和targets
        
        Args:
            q_embed (torch.Tensor): 查询特征，形状为[B, base_dim]
            k_embeds (List[torch.Tensor]): 键特征列表，每个元素形状为[B, base_dim]
            
        Returns:
            tuple: 包含两个列表的元组:
                - logits_list (List[torch.Tensor]): 每个视图的logits列表
                - targets_list (List[torch.Tensor]): 每个视图的目标标签列表
                
        Raises:
            ValueError: 当k_embeds的数量与num_views不匹配或维度不正确时抛出异常
        """
        # 前向：计算每个视图的对比 logits 与标签（正样本为第0列）
        if len(k_embeds) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(k_embeds)}")
        for k in k_embeds:
            if k.dim() != 2 or k.shape != q_embed.shape:
                raise ValueError("Each k_embed must be 2D and same shape as q_embed")

        # 归一化 q
        q = F.normalize(self.q_proj(q_embed), dim=1)
        # 步数自增
        self.global_step = int(self.global_step) + 1
        # 修正：当 queue_warmup_steps=0 时不应进入 warmup
        warmup = self.global_step < self.queue_warmup_steps

        logits_list, targets_list = [], []

        with torch.no_grad():
            self.momentum_update_key_encoders()

        for i, k_embed in enumerate(k_embeds):
            with torch.no_grad():
                k = F.normalize(self.k_projs[i](k_embed), dim=1)

            queue = getattr(self, f"queue_{i}")
            # 正样本
            l_pos = torch.sum(q * k, dim=1, keepdim=True)
            # 负样本
            if warmup:
                sim = torch.matmul(q, k.t())
                N = sim.size(0)
                if N > 1:
                    mask = ~torch.eye(N, dtype=torch.bool, device=sim.device)
                    l_neg = sim[mask].view(N, N - 1)
                else:
                    l_neg = torch.matmul(q, queue.clone().detach())
            else:
                l_neg = torch.matmul(q, queue.clone().detach())
            logits = torch.cat([l_pos, l_neg], dim=1) / self.T
            targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

            if self.debug and (self.global_step == 1 or self.global_step == self.queue_warmup_steps or self.global_step <= 3):
                with torch.no_grad():
                    cos_stats = {
                        "mean": float((q * k).sum(dim=1).mean().item()),
                        "std": float((q * k).sum(dim=1).std(unbiased=False).item()) if q.size(0) > 1 else 0.0,
                        "min": float((q * k).sum(dim=1).min().item()),
                        "max": float((q * k).sum(dim=1).max().item()),
                    }
                    qshape = list(queue.shape)
                    fill_ratio = (0.0 if warmup else float(min(self._filled[i], self.K)) / float(self.K))
                    assert int(targets.sum().item()) == 0, "MoCo targets 应全为 0"
                print(f"[EM.moco][multi][v={i}] step={self.global_step} warmup={warmup} q={list(q.shape)} k={list(k.shape)} logits={list(logits.shape)} queue={qshape} fill_ratio={fill_ratio:.2f} cos={cos_stats}")

            logits_list.append(logits)
            targets_list.append(targets)

            # 更新队列（非 warmup）
            if not warmup:
                self._dequeue_and_enqueue(k, i)
                self._filled[i] = int(min(self.K, int(self._filled[i]) + k.size(0)))

        return logits_list, targets_list


class MoCoV2MultiViewDoubleTau(nn.Module):
    """
    多视图 MoCo v2实现 (双温度参数版本)
    
    使用 τ₁ 控制类内吸引强度，τ₂ 控制类间排斥强度。
    
    Attributes:
        num_views (int): 视图数量
        K (int): 队列大小
        m (float): 动量更新系数
        tau1 (float): 正样本对温度系数 (类内吸引)
        tau2 (float): 负样本对温度系数 (类间排斥)
        queue_warmup_steps (int): 队列预热步数
        debug (bool): 是否启用调试模式
        global_step (int): 全局训练步数
        q_proj (nn.Sequential): 查询投影头网络
        k_projs (nn.ModuleList): 键投影头网络列表
        
    Methods:
        momentum_update_key_encoders: 动量更新所有键编码器
        _dequeue_and_enqueue: 将键特征入队到对应视图的队列中
        forward: 前向传播，计算对比损失的logits和targets
    """
    # 将 T 替换为 tau1 和 tau2，并给 tau2 一个默认值
    def __init__(self, base_dim: int, proj_dim: int, num_views: int, K: int = 4096, m: float = 0.999, tau1: float = 0.2, tau2: Union[float, None] = None, queue_warmup_steps: int = 0, debug: bool = False):
        """
        初始化多视图MoCo v2模型 (双温度参数版本)
        
        Args:
            base_dim (int): 基础维度，即输入特征的维度
            proj_dim (int): 投影维度，即投影头输出特征的维度
            num_views (int): 视图数量，必须大于等于1
            K (int, optional): 队列大小，默认为4096
            m (float, optional): 动量更新系数，默认为0.999
            tau1 (float, optional): 正样本对温度系数 (类内吸引)，默认为0.2
            tau2 (float or None, optional): 负样本对温度系数 (类间排斥)，
                                            若为None，则使用tau1的值。默认为None。
            queue_warmup_steps (int, optional): 队列预热步数，默认为0
            debug (bool, optional): 是否启用调试模式，默认为False
        """
        super().__init__()
        assert num_views >= 1, "num_views must be >= 1"
        assert proj_dim is not None and proj_dim > 0, "proj_dim must be positive"
        self.num_views = int(num_views)
        self.K = int(K)
        self.m = float(m)
        self.tau1 = float(tau1)
        # 如果 tau2 未指定，则使用 tau1 (兼容单温度参数情况)
        self.tau2 = float(tau2) if tau2 is not None else self.tau1 
        self.queue_warmup_steps = int(queue_warmup_steps)
        self.debug = bool(debug)
        self.global_step = 0
        self._filled = [0 for _ in range(self.num_views)]

        # 共享 q 投影头
        self.q_proj = nn.Sequential(
            nn.Linear(base_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        # 独立 k 投影头
        self.k_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_dim, proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim),
            ) for _ in range(self.num_views)
        ])
        # 初始化各 k_proj = q_proj，且冻结梯度
        with torch.no_grad():
            for k_proj in self.k_projs:
                for qp, kp in zip(self.q_proj.parameters(), k_proj.parameters()):
                    kp.data.copy_(qp.data)
                    kp.requires_grad = False

        # 为每个视图注册独立队列与指针
        for i in range(self.num_views):
            self.register_buffer(f"queue_{i}", F.normalize(torch.randn(proj_dim, self.K), dim=0))
            self.register_buffer(f"queue_ptr_{i}", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update_key_encoders(self):
        """
        使用动量更新策略更新所有键编码器的参数
        """
        # 动量更新：将 k 编码器向 q 编码器移动
        for k_proj in self.k_projs:
            for param_q, param_k in zip(self.q_proj.parameters(), k_proj.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor, view_idx: int):
        """
        将键特征入队到指定视图的队列中
        """
        # 出队入队：环形队列更新（安全切片，避免越界）
        keys = keys.detach()
        n_new = int(keys.shape[0])
        if n_new <= 0:
            return
        queue = getattr(self, f"queue_{view_idx}")
        queue_ptr = getattr(self, f"queue_ptr_{view_idx}")
        K = int(queue.size(1))  # 队列列数作为容量
        ptr = int(queue_ptr.item())
        # 统一用转置后的 [C, B] 视图进行列区间写入
        kT = keys.t()  # [C, n_new]

        # 剩余容量
        rem = K - ptr
        if n_new <= rem:
            # 单段写入：完全装入尾部
            queue[:, ptr:ptr + n_new] = kT[:, :n_new]
            ptr = (ptr + n_new) % K
        else:
            # 两段写入：尾段 + 头段
            len1 = rem
            if len1 > 0:
                queue[:, ptr:ptr + len1] = kT[:, :len1]
            len2 = min(n_new - len1, K)  # 头段长度不超过 K
            if len2 > 0:
                queue[:, 0:len2] = kT[:, len1:len1 + len2]
            ptr = (ptr + n_new) % K

        queue_ptr[0] = ptr

    def forward(self, q_embed: torch.Tensor, k_embeds: List[torch.Tensor]):
        """
        前向传播函数
        
        计算每个视图的对比学习logits和targets
        """
        # ... (参数检查代码保持不变) ...
        if len(k_embeds) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(k_embeds)}")
        for k in k_embeds:
            if k.dim() != 2 or k.shape != q_embed.shape:
                raise ValueError("Each k_embed must be 2D and same shape as q_embed")

        # 归一化 q
        q = F.normalize(self.q_proj(q_embed), dim=1)
        # 步数自增
        self.global_step = int(self.global_step) + 1
        # 修正：当 queue_warmup_steps=0 时不应进入 warmup
        warmup = self.global_step < self.queue_warmup_steps

        logits_list, targets_list = [], []

        with torch.no_grad():
            self.momentum_update_key_encoders()

        for i, k_embed in enumerate(k_embeds):
            with torch.no_grad():
                k = F.normalize(self.k_projs[i](k_embed), dim=1)

            queue = getattr(self, f"queue_{i}")
            
            # --- V2.0 双温度参数修改区域 START ---
            
            # 正样本相似度
            l_pos = torch.sum(q * k, dim=1, keepdim=True)
            # 负样本相似度
            if warmup:
                sim = torch.matmul(q, k.t())
                N = sim.size(0)
                if N > 1:
                    mask = ~torch.eye(N, dtype=torch.bool, device=sim.device)
                    # 队列预热阶段，负样本为当前 batch 内的其他样本
                    l_neg = sim[mask].view(N, N - 1) 
                else:
                    # 如果 batch_size=1，或者不需要 batch 内负样本，则退回到队列负样本
                    l_neg = torch.matmul(q, queue.clone().detach())
            else:
                # 队列负样本
                l_neg = torch.matmul(q, queue.clone().detach())
                
            # 分别除以 τ₁ 和 τ₂
            l_pos_scaled = l_pos / self.tau1
            l_neg_scaled = l_neg / self.tau2
            
            # 拼接 logits
            logits = torch.cat([l_pos_scaled, l_neg_scaled], dim=1)
            # 目标标签 (正样本在第0列)
            targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            
            # --- V2.0 双温度参数修改区域 END ---

            if self.debug and (self.global_step == 1 or self.global_step == self.queue_warmup_steps or self.global_step <= 3):
                with torch.no_grad():
                    cos_stats = {
                        "mean": float((q * k).sum(dim=1).mean().item()),
                        "std": float((q * k).sum(dim=1).std(unbiased=False).item()) if q.size(0) > 1 else 0.0,
                        "min": float((q * k).sum(dim=1).min().item()),
                        "max": float((q * k).sum(dim=1).max().item()),
                    }
                    qshape = list(queue.shape)
                    fill_ratio = (0.0 if warmup else float(min(self._filled[i], self.K)) / float(self.K))
                    assert int(targets.sum().item()) == 0, "MoCo targets 应全为 0"
                    print(f"[EM.moco][multi][v={i}] step={self.global_step} warmup={warmup} q={list(q.shape)} k={list(k.shape)} logits={list(logits.shape)} queue={qshape} fill_ratio={fill_ratio:.2f} tau1={self.tau1} tau2={self.tau2} cos={cos_stats}")

            logits_list.append(logits)
            targets_list.append(targets)

            # 更新队列（非 warmup）
            if not warmup:
                self._dequeue_and_enqueue(k, i)
                self._filled[i] = int(min(self.K, int(self._filled[i]) + k.size(0)))

        return logits_list, targets_list

# =================================================
# BYOL Loss Functions (BYOL损失函数)
# =================================================

class BYOLLoss(nn.Module):
    """
    BYOL损失函数模块
    
    实现BYOL算法的核心损失函数，基于对称余弦相似度损失。
    该损失函数不需要负样本，通过在线网络预测和目标网络输出之间的
    对称对比来学习表示。
    """
    
    def __init__(self, temperature: float = 0.2):
        """
        初始化BYOL损失函数
        
        Args:
            temperature: 温度参数，用于缩放余弦相似度
        """
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
    
    def forward(self, online_view1: torch.Tensor, online_view2: torch.Tensor, 
                target_view1: torch.Tensor, target_view2: torch.Tensor) -> torch.Tensor:
        """
        计算BYOL对称损失
        
        Args:
            online_view1: 在线网络第一个视图的预测，形状为[B, D]
            online_view2: 在线网络第二个视图的预测，形状为[B, D]  
            target_view1: 目标网络第一个视图的输出，形状为[B, D]
            target_view2: 目标网络第二个视图的输出，形状为[B, D]
            
        Returns:
            torch.Tensor: BYOL对称损失值
        """
        # 归一化所有表示
        online_view1 = F.normalize(online_view1, dim=1)
        online_view2 = F.normalize(online_view2, dim=1)
        target_view1 = F.normalize(target_view1, dim=1)
        target_view2 = F.normalize(target_view2, dim=1)
        
        # 计算对称损失（使用temperature缩放）
        cosine_sim_1_to_2 = self.cosine_similarity(online_view1, target_view2.detach()).mean()
        cosine_sim_2_to_1 = self.cosine_similarity(online_view2, target_view1.detach()).mean()
        
        loss_1_to_2 = (2 - 2 * cosine_sim_1_to_2) / self.temperature
        loss_2_to_1 = (2 - 2 * cosine_sim_2_to_1) / self.temperature
        
        return (loss_1_to_2 + loss_2_to_1) / 2

def compute_byol_loss(online_predictions: List[torch.Tensor], 
                      target_outputs: List[torch.Tensor],
                      temperature: float = 0.2) -> torch.Tensor:
    """
    计算多视图BYOL损失的便利函数
    
    Args:
        online_predictions: 在线网络预测输出列表
        target_outputs: 目标网络输出列表  
        temperature: 温度参数
        
    Returns:
        torch.Tensor: BYOL损失值
    """
    if len(online_predictions) < 2 or len(target_outputs) < 2:
        return torch.tensor(0.0)
    
    losses = []
    
    # 计算所有视图对之间的对称损失
    for i in range(len(online_predictions)):
        for j in range(len(target_outputs)):
            if i != j:
                # 归一化
                online_pred = F.normalize(online_predictions[i], dim=1)
                target_out = F.normalize(target_outputs[j], dim=1)
                
                # 计算余弦相似度损失（使用temperature缩放）
                cosine_sim = (online_pred * target_out).sum(dim=1).mean()
                loss = (2 - 2 * cosine_sim) / temperature
                losses.append(loss)
    
    # 返回平均损失
    return torch.stack(losses).mean() if losses else torch.tensor(0.0)

# =================================================
# BYOL Multi-View (多视图BYOL模型)
# =================================================

class BYOLMultiView(nn.Module):
    """
    正统多视图BYOL (Bootstrap Your Own Latent) 模型
    
    核心设计原则：
    1. 单一共享编码器 - 所有视图共享同一个encoder参数
    2. 单一预测头 - 所有视图共享同一个predictor参数  
    3. 多视图来自数据增强 - 而非网络结构差异
    4. EMA更新在optimizer.step()之后 - 而非forward中
    """
    
    def __init__(self, base_dim: int, proj_dim: int, num_views: int, 
                 predictor_dim: int = 256, m: float = 0.996, temperature: float = 0.2,
                 debug: bool = False):
        """
        初始化正统BYOL多视图模型
        
        Args:
            base_dim: 输入特征维度
            proj_dim: 投影头输出维度
            num_views: 视图数量（用于数据增强）
            predictor_dim: 预测头维度
            m: 指数移动平均(EMA)系数
            temperature: 温度参数，用于缩放余弦相似度损失
            debug: 调试模式
        """
        super().__init__()
        
        self.base_dim = base_dim
        self.proj_dim = proj_dim
        self.num_views = num_views
        self.predictor_dim = predictor_dim
        self.m = m
        self.temperature = temperature
        self.debug = debug
        
        # ✅ 正统设计：单一共享编码器（所有视图共享参数）
        self.online_encoder = self._build_projector(base_dim, proj_dim)
        
        # ✅ 正统设计：单一共享预测头（所有视图共享参数）
        self.predictor = self._build_predictor(proj_dim, predictor_dim)
        
        # ✅ 正统设计：单一目标编码器（EMA更新的教师网络）
        self.target_encoder = self._build_projector(base_dim, proj_dim)
        
        # 初始化目标网络权重（复制在线网络权重）
        self._init_target_encoder()
        
        # 停止目标网络的梯度 - 不参与梯度计算
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # 调试标志
        self._update_called = False  # 用于检测EMA更新时机
    
    def _build_projector(self, input_dim: int, output_dim: int) -> nn.Sequential:
        """
        构建投影头（BYOL论文标准架构）
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            
        Returns:
            nn.Sequential: 投影头网络
        """
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )
    
    def _build_predictor(self, input_dim: int, output_dim: int) -> nn.Sequential:
        """
        构建预测头（BYOL论文标准架构）
        
        Args:
            input_dim: 输入维度（投影输出维度）
            output_dim: 预测头隐藏维度
            
        Returns:
            nn.Sequential: 预测头网络
        """
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, input_dim),  # 输出维度回到投影维度
        )
    
    def _init_target_encoder(self):
        """初始化目标网络权重（完全复制在线网络权重）"""
        with torch.no_grad():
            for target_param, online_param in zip(self.target_encoder.parameters(), 
                                          self.online_encoder.parameters()):
                target_param.data.copy_(online_param.data)
    
    @torch.no_grad()
    def update_target_encoder(self):
        """
        ✅ 正统EMA更新：在optimizer.step()之后调用
        
        EMA公式: target_param = m * target_param + (1 - m) * online_param
        """
        with torch.no_grad():
            for target_param, online_param in zip(self.target_encoder.parameters(), 
                                          self.online_encoder.parameters()):
                target_param.data = self.m * target_param.data + (1.0 - self.m) * online_param.data
        
        if self.debug:
            print(f"[BYOL] EMA更新完成，动量系数={self.m}")
    
    def forward(self, views: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        ✅ 正统前向传播：多视图共享编码器
        
        Args:
            views: 输入视图列表，每个视图形状为 [batch_size, base_dim]
                  视图差异来自数据增强，而非网络参数
            
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: 
                (在线网络预测列表, 目标网络输出列表)
        """
        assert len(views) == self.num_views, f"输入视图数量({len(views)})与模型配置({self.num_views})不匹配"
        
        online_predictions = []
        target_outputs = []
        
        # ✅ 正统设计：所有视图共享同一个编码器
        for i, view in enumerate(views):
            # 在线网络路径：编码 -> 预测
            online_encoded = self.online_encoder(view)  # 共享编码器
            online_prediction = self.predictor(online_encoded)  # 共享预测头
            
            # 目标网络路径：编码（停止梯度）
            with torch.no_grad():
                target_encoded = self.target_encoder(view)  # 共享目标编码器
            
            online_predictions.append(online_prediction)
            target_outputs.append(target_encoded)
        
        if self.debug:
            print(f"[BYOL] Forward: 处理了{len(views)}个视图，编码器参数共享")
        
        # ❌ 错误：不在forward中更新target！
        # EMA更新应该在optimizer.step()之后通过update_target_encoder()调用
        
        return online_predictions, target_outputs
    
    def get_loss(self, online_predictions: List[torch.Tensor], 
                 target_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        ✅ 正统BYOL对称损失计算
        
        计算策略：
        1. 每个在线预测与所有其他目标输出计算损失
        2. 使用余弦相似度作为距离度量
        3. 对称性确保表示一致性
        
        Args:
            online_predictions: 在线网络预测输出列表
            target_outputs: 目标网络输出列表
            
        Returns:
            torch.Tensor: BYOL对称损失值
        """
        losses = []
        
        # ✅ 正统损失：多视图对称对比
        # 对于每个视图i，计算其在线预测与所有其他视图j(j≠i)的目标输出之间的损失
        for i in range(len(online_predictions)):
            for j in range(len(target_outputs)):
                if i != j:
                    # L2归一化（BYOL标准做法）
                    online_pred = F.normalize(online_predictions[i], dim=1)
                    target_out = F.normalize(target_outputs[j], dim=1)
                    
                    # BYOL损失：(2 - 2 * cosine_similarity) / temperature
                    # 等价于 (1 - cosine_similarity) * 2 / temperature
                    cosine_sim = (online_pred * target_out).sum(dim=1)
                    loss = (2 - 2 * cosine_sim.mean()) / self.temperature
                    losses.append(loss)
        
        if self.debug:
            print(f"[BYOL] Loss: 计算了{len(losses)}个视图对损失，均值={torch.stack(losses).mean().item():.6f}")
        
        # 返回平均损失
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=online_predictions[0].device)
    
    def get_training_info(self) -> dict:
        """
        获取BYOL模型训练信息（用于调试和验证）
        
        Returns:
            dict: 包含模型架构和状态信息的字典
        """
        return {
            "architecture": "正统BYOL多视图",
            "shared_encoder": True,
            "shared_predictor": True,
            "ema_momentum": self.m,
            "num_views": self.num_views,
            "proj_dim": self.proj_dim,
            "predictor_dim": self.predictor_dim,
            "online_encoder_params": sum(p.numel() for p in self.online_encoder.parameters()),
            "target_encoder_params": sum(p.numel() for p in self.target_encoder.parameters()),
            "predictor_params": sum(p.numel() for p in self.predictor.parameters()),
            "update_called": self._update_called
        }


# =================================================
# Enhanced MoCo V2 Multi-View (高级版本)
# =================================================



# =================================================
# EM 主模型（组合：编码器 + 融合 + MoCo）
# =================================================
class EM(nn.Module):
    """
    EM主模型类，组合了编码器、融合解码器和MoCo多视图对比学习模块
    增强版：支持两实体和三实体融合模式
    
    该模型使用GAT-GT串行编码器进行节点表示学习，通过图变换器风格的融合解码器进行实体关系预测，
    并采用MoCo多视图对比学习框架进行自监督预训练。支持两实体和三实体融合模式。
    
    模型主要组件：
    - 编码器：GATGTSerial，先GATConv再TransformerConv的串联编码器
    - 融合解码器：FusionDecoder（两实体）或TriFusionDecoder（三实体）
    - 对比学习：MoCoV2MultiView，多视图动量对比学习框架
    - 对抗分支：用于节点级对抗训练的线性头
    
    Attributes:
        encoder (GATGTSerial): 图神经网络编码器
        read (AvgReadout): 图级读出函数
        mlp1 (nn.Linear): 图级表示的MLP变换层
        sigm (nn.Sigmoid): Sigmoid激活函数
        moco (MoCoV2MultiView): 多视图MoCo对比学习模块
        fusion (Union[FusionDecoder, TriFusionDecoder]): 融合解码器（两实体或三实体）
        tri_fusion (TriFusionDecoder): 三实体融合解码器（仅在三实体模式下使用）
        is_tri_entity (bool): 是否使用三实体模式
        adv_head (nn.Linear): 对抗训练分支
        dropout (float): Dropout概率
        aug_list (List[str]): 数据增强方法列表
        noise_std (float): 噪声增强标准差
        mask_rate (float): 掩码增强比例
        base_seed (int): 增强随机种子基础值
    """
    def __init__(self, feature: int, hidden1: int, hidden2: int, decoder1: int, dropout: float, 
                 tri_entity_mode: bool = False):
        """
        初始化EM模型
        
        Args:
            feature (int): 输入特征维度
            hidden1 (int): 编码器第一层隐藏维度
            hidden2 (int): 编码器第二层隐藏维度
            decoder1 (int): 解码器第一层维度
            dropout (float): Dropout概率
        """
        super().__init__()
        # 编码器
        gat_heads = int(getattr(args, "gat_heads", 4) or 4)
        self.encoder = GATGTSerial(in_dim=feature, hidden1=hidden1, hidden2=hidden2, dropout=dropout, gat_heads=gat_heads)

        # 读出与 MLP
        self.read = AvgReadout()
        self.mlp1 = nn.Linear(hidden2, hidden2)
        self.sigm = nn.Sigmoid()

        # 自监督学习模块（MoCo或BYOL）
        proj_dim = int(getattr(args, "proj_dim", hidden2) or hidden2)
        num_views = int(getattr(args, "num_views", 3) or 3)
        self.enable_view_0 = bool(getattr(args, "enable_view_0", True))
        # 根据是否启用第0视图计算实际视图数量
        self.actual_num_views = num_views if self.enable_view_0 else max(0, num_views - 1)
        
        # 获取模型类型
        model_type = getattr(args, "model_type", "moco")
        
        # 使用工厂模式创建自监督学习模块
        try:
            if model_type == "byol":
                # 创建BYOL模型
                byol_config = getattr(args, "byol_config", None)
                if byol_config:
                    # 使用高级配置字符串
                    self.ssl_module = ModelFactory.create_model(
                        model_type="byol",
                        base_dim=hidden2,
                        proj_dim=proj_dim,
                        num_views=max(1, self.actual_num_views),
                        config_str=byol_config
                    )
                    ssl_type = byol_config.split('[')[0]
                else:
                    # 使用基础配置
                    ssl_type = "basic"
                    self.ssl_module = ModelFactory.create_model(
                        model_type="byol",
                        base_dim=hidden2,
                        proj_dim=proj_dim,
                        num_views=max(1, self.actual_num_views),
                        config_str=ssl_type,
                        predictor_dim=int(getattr(args, "byol_predictor_dim", 256)),
                        m=float(getattr(args, "byol_ema_momentum", 0.996))
                    )
            else:
                # 创建MoCo模型（默认）
                moco_config = getattr(args, "moco_config", None)
                if moco_config:
                    # 使用高级配置字符串
                    self.ssl_module = ModelFactory.create_model(
                        model_type="moco",
                        base_dim=hidden2,
                        proj_dim=proj_dim,
                        num_views=max(1, self.actual_num_views),
                        config_str=moco_config,
                        queue_warmup_steps=int(getattr(args, "queue_warmup_steps", 0)),
                        debug=bool(getattr(args, "moco_debug", False))
                    )
                    ssl_type = moco_config.split('[')[0]
                else:
                    # 使用基础配置
                    ssl_type = getattr(args, "moco_type", "basic")
                    self.ssl_module = ModelFactory.create_model(
                        model_type="moco",
                        base_dim=hidden2,
                        proj_dim=proj_dim,
                        num_views=max(1, self.actual_num_views),
                        config_str=ssl_type,
                        K=int(getattr(args, "moco_K", 4096)),
                        m=float(getattr(args, "moco_m", 0.999)),
                        T=float(getattr(args, "moco_T", 0.2)),
                        queue_warmup_steps=int(getattr(args, "queue_warmup_steps", 0)),
                        debug=bool(getattr(args, "moco_debug", False)),
                        # DoubleTau版本特有参数（仅当使用double_tau时有效）
                        tau1=float(getattr(args, "moco_tau1", 0.2)),
                        tau2=float(getattr(args, "moco_tau2", 0.3))
                    )
        except Exception as e:
            print(f"Warning: Failed to create {model_type} model with factory method: {e}")
            print("Falling back to manual configuration...")
            
            # 回退到手动配置
            if model_type == "byol":
                ssl_type = getattr(args, "byol_type", "basic")
                byol_base_args = {
                    "base_dim": hidden2,
                    "proj_dim": proj_dim,
                    "num_views": max(1, self.actual_num_views),
                    "predictor_dim": int(getattr(args, "byol_predictor_dim", 256)),
                    "m": float(getattr(args, "byol_ema_momentum", 0.996)),
                    "temperature": float(getattr(args, "byol_temperature", 0.2)),
                    "debug": bool(getattr(args, "moco_debug", False)),
                }
                self.ssl_module = BYOLMultiView(**byol_base_args)
            else:
                ssl_type = getattr(args, "moco_type", "basic")
                moco_base_args = {
                    "base_dim": hidden2,
                    "proj_dim": proj_dim,
                    "num_views": max(1, self.actual_num_views),
                    "K": int(getattr(args, "moco_K", 4096)),
                    "m": float(getattr(args, "moco_m", 0.999)),
                    "base_T": float(getattr(args, "moco_T", 0.2)),
                    "queue_warmup_steps": int(getattr(args, "queue_warmup_steps", 0)),
                    "debug": bool(getattr(args, "moco_debug", False)),
                }
                
                # 标准版本使用T参数而不是base_T
                standard_args = moco_base_args.copy()
                standard_args["T"] = standard_args.pop("base_T")
                self.ssl_module = MoCoV2MultiView(**standard_args)
        
        # 保存模型类型信息，用于可能的调试或日志
        self.model_type = model_type
        self.ssl_type = ssl_type

        # 初始化增强相关参数
        self.base_seed = int(getattr(args, "seed", 0))
        # 从--augment参数获取增强列表，处理字符串或列表两种情况
        augment_param = getattr(args, "augment", "random_permute_features,attribute_mask,noise_then_mask")
        if isinstance(augment_param, str):
            self.aug_list = [aug.strip() for aug in augment_param.split(",") if aug.strip()]
        elif isinstance(augment_param, list):
            self.aug_list = augment_param
        else:
            self.aug_list = ["random_permute_features", "attribute_mask", "noise_then_mask"]
        self.noise_std = float(getattr(args, "noise_std", 0.1))
        self.mask_rate = float(getattr(args, "mask_rate", 0.1))

        # 初始化对抗训练分支
        self.adv_head = nn.Linear(hidden2, 1)

        # 智能注意力配置决策
        fusion_strategy, fusion_kwargs = self._determine_fusion_config()
        
        heads = int(getattr(args, "fusion_heads", 4) or 4)
        self.fusion = FusionDecoder(
            hidden_dim=hidden2, 
            decoder1_dim=decoder1, 
            heads=heads, 
            dropout=dropout,
            fusion_strategy=fusion_strategy,
            **fusion_kwargs
        )
    
    def _determine_fusion_config(self) -> tuple[str, dict]:
        """
        智能确定融合策略配置
        根据命令行参数自动选择最佳的注意力机制配置
        
        Returns:
            tuple[str, dict]: (融合策略字符串, 参数字典)
        """
        # 检查是否指定了高级配置
        attention_config = getattr(args, "attention_config", None)
        if attention_config:
            return attention_config, {}
        
        # 检查基础配置
        use_co_attention = getattr(args, "use_co_attention", False)
        use_multihead = getattr(args, "use_multihead", False)
        
        fusion_strategy = getattr(args, "fusion_strategy", "self_attention") or "self_attention"
        fusion_kwargs = {}
        
        if use_multihead:
            # 多头场景
            if transformer_style:
                fusion_strategy = f"transformer_multihead[num_heads={getattr(args, 'fusion_heads', 4)}]"
            else:
                fusion_strategy = "multihead"
        
        elif use_co_attention:
            # 协作注意力场景
            co_attention_type = getattr(args, "co_attention_type", "transformer")
            fusion_strategy = "co_attention"
            fusion_kwargs["co_attention_type"] = co_attention_type
            co_hidden_dim = getattr(args, "co_hidden_dim", None)
            if co_hidden_dim:
                fusion_kwargs["co_hidden_dim"] = co_hidden_dim
        
        # 处理混合策略的特殊参数
        elif fusion_strategy == "hybrid":
            fusion_weight = getattr(args, "fusion_weight", 0.5)
            fusion_kwargs["fusion_weight"] = fusion_weight
            co_attention_type = getattr(args, "co_attention_type", "transformer")
            fusion_kwargs["co_attention_type"] = co_attention_type
            co_hidden_dim = getattr(args, "co_hidden_dim", None)
            if co_hidden_dim:
                fusion_kwargs["co_hidden_dim"] = co_hidden_dim
        
        # 处理协作注意力融合的特殊参数
        elif fusion_strategy == "co_attention":
            co_hidden_dim = getattr(args, "co_hidden_dim", None)
            if co_hidden_dim:
                fusion_kwargs["co_hidden_dim"] = co_hidden_dim
            co_attention_type = getattr(args, "co_attention_type", "transformer")
            fusion_kwargs["co_attention_type"] = co_attention_type
        
        return fusion_strategy, fusion_kwargs

    def forward(self, data_o, data_a, idx):
        """
        前向传播函数(预测计算)
        
        Args:
            data_o: 原始图数据批次
            data_a: 增强图数据批次
            idx: 实体索引用于融合解码
            
        Returns:
            tuple: 包含以下元素的元组：
                - log (torch.Tensor): [B,1] 关联预测主任务输出
                - cla_os (torch.Tensor): MoCo对比logits（第0视图）
                - cla_os_a (torch.Tensor): MoCo对比targets（第0视图）
                - x2_o (torch.Tensor): [N,H] 原图节点表示（共享编码器输出）
                - logits_adv (torch.Tensor): [1,2N] 节点级对抗二分类logits
                - log1 (torch.Tensor): [B,decoder1] 融合解码器中间层输出
        """
        # data_o: 原图 batch；data_a: 损图 batch；idx: 实体索引用于融合解码
        x_o, edge_index = data_o.x, data_o.edge_index
        x_a = data_a.x
        if edge_index.device != x_o.device:
            edge_index = edge_index.to(x_o.device)

        # 编码原/损图
        x2_o = self.encoder.encode(x_o, edge_index)
        x2_o_a = self.encoder.encode(x_a, edge_index)

        # 图级表示
        h_os = self.sigm(self.read(x2_o))
        h_os = self.mlp1(h_os)
        h_os_a = self.sigm(self.read(x2_o_a))
        h_os_a = self.mlp1(h_os_a)

        # 多视图自监督学习：第0视图用损图，其余来自原图的增强
        num_views = int(getattr(args, "num_views", 3) or 3)
        
        k_embeds: List[torch.Tensor] = []
        if self.enable_view_0:
            k_embeds.append(x2_o_a)
        
        # 使用初始化时计算的实际视图数量
        # 调试信息：打印关键参数
        # print(f"DEBUG: num_views={num_views}, enable_view_0={self.enable_view_0}, actual_num_views={self.actual_num_views}")
        # print(f"DEBUG: Before loop, k_embeds length={len(k_embeds)}")
        # 修复：循环应该生成足够的视图以满足模型期望
        target_views = self.ssl_module.num_views
        current_views = len(k_embeds)
        for vid in range(1, target_views - current_views + 1):
            # print(f"DEBUG: Loop iteration vid={vid}")
            seed_v = self.base_seed + vid
            aug_name = self.aug_list[(vid - 1) % len(self.aug_list)]
            x_aug = apply_augmentation(
                aug_name, x_o, noise_std=self.noise_std, mask_rate=self.mask_rate, seed=seed_v
            )
            x2_aug = self.encoder.encode(x_aug, edge_index)
            k_embeds.append(x2_aug)
            
        # print(f"DEBUG: After loop, k_embeds length={len(k_embeds)}")
        # print(f"DEBUG: Model expects {self.ssl_module.num_views} views")
        
        # 根据模型类型处理自监督学习输出
        if self.model_type == "byol":
            # BYOL模型：返回在线预测和目标输出
            online_predictions, target_outputs = self.ssl_module(k_embeds)
            cla_os = online_predictions[0] if len(online_predictions) > 0 else None
            cla_os_a = target_outputs[0] if len(target_outputs) > 0 else None
        else:
            # MoCo模型：返回logits和targets
            logits_list, targets_list = self.ssl_module(x2_o, k_embeds)
            cla_os = logits_list[0] if len(logits_list) > 0 else None
            cla_os_a = targets_list[0] if len(targets_list) > 0 else None

        # 两实体模式：保持原有逻辑兼容
        if args.task_type == 'LDA':
            entity1 = x2_o[idx[0]]
            entity2 = x2_o[idx[1] + 240]
        elif args.task_type == 'MDA':
            entity1 = x2_o[idx[0] + 645]
            entity2 = x2_o[idx[1] + 240]
        elif args.task_type == 'LMI':
            entity1 = x2_o[idx[0]]
            entity2 = x2_o[idx[1] + 645]
        else:
            # 保守兜底：直接按给定索引
            entity1 = x2_o[idx[0]]
            entity2 = x2_o[idx[1]]

        # 两实体融合解码
        log, log1 = self.fusion(entity1, entity2)

        # 对抗 logits（沿特征求和）
        sc_1 = self.adv_head(x2_o).sum(1).unsqueeze(0)
        sc_2 = self.adv_head(x2_o_a).sum(1).unsqueeze(0)
        logits_adv = torch.cat((sc_1, sc_2), 1)

        return log, cla_os, cla_os_a, x2_o, logits_adv, log1

# =================================================
# 数据标注/三元组构建
# =================================================
def load_positive(in_file: str, seed: int):
    """
    该函数用于加载正样本数据，并使用指定的随机种子对其进行打乱处理。
    正样本通常表示已知存在的关联关系，例如药物-疾病关联等。
    
    参数:
        in_file (str): 正样本数据文件路径
        seed (int): 随机种子，用于保证实验可复现性
        
    返回:
        np.ndarray: 打乱后的正样本数据数组，形状为 (N, 2)，其中 N 是样本数量，
                   每一行包含两个实体的索引，表示它们之间存在关联关系
    """
    positive = np.loadtxt(em_path(in_file), dtype=np.int64)
    link_size = int(positive.shape[0])  # 保留全部
    rng = np.random.default_rng(int(seed))  # 使用局部生成器，避免污染全局随机状态
    idx = rng.permutation(positive.shape[0])
    positive = positive[idx]
    positive = positive[:link_size]
    return positive

def load_negative_all(neg_file: str, seed: int):
    """
    读取并打乱负样本全集（未知关联），返回数组 shape=(M, 2)
    
    该函数用于加载所有负样本数据（即未知关联的实体对），并使用指定的随机种子对其进行
    随机打乱处理，以确保实验的可重复性。
    
    Args:
        neg_file (str): 负样本数据文件路径
        seed (int): 随机种子，用于初始化随机数生成器以保证结果可重现
        
    Returns:
        np.ndarray: 打乱后的负样本数据数组，形状为(M, 2)，其中M是负样本数量，
                   每一行包含两个实体的索引，表示它们之间不存在关联关系
    """
    negative_all = np.loadtxt(em_path(neg_file), dtype=np.int64)
    rng = np.random.default_rng(int(seed))  # 使用局部生成器，避免污染全局随机状态
    idx = rng.permutation(negative_all.shape[0])
    negative_all = negative_all[idx]
    return negative_all

def sample_negative(negative_all: np.ndarray, pos_count: int):
    """
    该函数从负样本全集中采样与正样本数量相等的负样本，用于构建平衡的数据集。
    采样方式为直接取前pos_count个样本，不进行随机采样。
    
    参数:
        negative_all (np.ndarray): 负样本全集，形状为 (M, 2)，每一行代表一个负样本对
        pos_count (int): 需要采样的负样本数量，通常等于正样本的数量
        
    返回:
        np.ndarray: 采样得到的负样本，形状为 (pos_count, 2)
        
    异常:
        ValueError: 当负样本全集数量不足时抛出异常
    """
    # 检查负样本全集是否足够
    if negative_all.shape[0] < pos_count:
        raise ValueError(f"负样本全集数量不足：需要 {pos_count}，实际 {negative_all.shape[0]}")
    # 直接取前pos_count个负样本
    negative = np.asarray(negative_all[:pos_count])
    return negative

def attach_labels(positive: np.ndarray, negative: np.ndarray, negative_all: np.ndarray):
    """
    为正/负样本分别附加标签列，输出：
    - positive_labeled: [i, j, 1]
    - negative_labeled: [i, j, 0]（采样得到，用于训练/测试）
    - negative_all_labeled: [i, j, 0]（全集，供需要时参考）
    
    该函数为正负样本添加标签列，正样本标记为1，负样本标记为0。
    返回三个数据集：带标签的正样本、带标签的负样本以及带标签的负样本全集。
    
    参数:
        positive (np.ndarray): 正样本数据，形状为 (N, 2)
        negative (np.ndarray): 负样本数据，形状为 (M, 2)
        negative_all (np.ndarray): 负样本全集，形状为 (K, 2)
        
    返回:
        tuple: 包含以下三个元素的元组：
            - positive_labeled (np.ndarray): 带标签的正样本，形状为 (N, 3)
            - negative_labeled (np.ndarray): 带标签的负样本，形状为 (M, 3)
            - negative_all_labeled (np.ndarray): 带标签的负样本全集，形状为 (K, 3)
    """
    # 创建正样本标签列，值全为1
    pos_lbl = np.ones(positive.shape[0], dtype=np.int64).reshape(positive.shape[0], 1)
    # 创建负样本标签列，值全为0
    neg_lbl = np.zeros(negative.shape[0], dtype=np.int64).reshape(negative.shape[0], 1)
    # 创建负样本全集标签列，值全为0
    neg_all_lbl = np.zeros(negative_all.shape[0], dtype=np.int64).reshape(negative_all.shape[0], 1)

    # 将正样本与其标签列拼接
    positive_labeled = np.concatenate([positive, pos_lbl], axis=1)
    # 将负样本与其标签列拼接
    negative_labeled = np.concatenate([negative, neg_lbl], axis=1)
    # 将负样本全集与其标签列拼接
    negative_all_labeled = np.concatenate([negative_all, neg_all_lbl], axis=1)

    return positive_labeled, negative_labeled, negative_all_labeled

def kfold_split_triples(positive_labeled: np.ndarray,
                        negative_labeled: np.ndarray,
                        k_fold: int = 5):
    """
    该函数实现K折交叉验证的数据划分，确保正负样本在每折中保持一致的划分方式。
    每一折中，测试集包含特定区间的正负样本，训练集包含其余的正负样本。
    
    参数:
        positive_labeled (np.ndarray): 带标签的正样本，形状为 (N, 3)
        negative_labeled (np.ndarray): 带标签的负样本，形状为 (N, 3)
        k_fold (int, optional): 折数，默认为5
        
    返回:
        tuple: 包含以下两个元素的元组：
            - train_data_folds (list): 训练集列表，每个元素形状为 (M, 3)
            - test_data_folds (list): 测试集列表，每个元素形状为 (M, 3)
            
    异常:
        ValueError: 当k_fold不是正整数或正负样本数量不一致时抛出异常
    """
    # 检查k_fold参数是否有效
    if k_fold <= 0:
        raise ValueError("k_fold 必须为正整数")
    # 检查正负样本数量是否一致
    if positive_labeled.shape[0] != negative_labeled.shape[0]:
        raise ValueError("正负样本数量必须一致以进行等量划分")

    # 获取正样本数量
    pos_num = positive_labeled.shape[0]
    # 计算每折的样本数量
    fold_size = pos_num // k_fold
    # 初始化训练集和测试集列表
    train_data_folds = []
    test_data_folds = []

    # 遍历每一折
    for fold in range(k_fold):
        # 计算当前折的起始和结束索引
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < k_fold - 1 else pos_num

        # 划分阳性样本为测试集和训练集
        test_positive = positive_labeled[start_idx:end_idx]
        train_positive = np.vstack((positive_labeled[:start_idx], positive_labeled[end_idx:]))

        # 划分阴性样本为测试集和训练集
        test_negative = negative_labeled[start_idx:end_idx]
        train_negative = np.vstack((negative_labeled[:start_idx], negative_labeled[end_idx:]))

        # 构建训练集和测试集：将正负样本合并
        train_data = np.vstack((train_positive, train_negative))
        test_data = np.vstack((test_positive, test_negative))

        # 将训练集和测试集添加到列表中
        train_data_folds.append(train_data)
        test_data_folds.append(test_data)

    return train_data_folds, test_data_folds

def build_triples(in_file: str,
                  neg_file: str,
                  seed: int = 0,
                  k_fold: int = 5):
    """
    主流程：构建样本三元组并进行五折划分
    与 CSGLMD-main/data_preprocess.py 的样本处理逻辑完全一致
    
    该函数负责加载正负样本数据，进行平衡采样，并将数据划分为K折交叉验证格式。
    每一折都包含训练集和测试集，用于机器学习模型的训练和评估。
    
    参数:
        in_file (str): 正样本数据文件路径
        neg_file (str): 负样本数据文件路径
        seed (int, optional): 随机种子，用于保证实验可复现性，默认为0
        k_fold (int, optional): K折交叉验证的折数，默认为5
        
    返回:
        tuple: 包含以下四个元素的元组：
            - train_data_folds (list[np.ndarray]): 每折训练三元组列表，每个元素形状为[N, 3]
            - test_data_folds (list[np.ndarray]): 每折测试三元组列表，每个元素形状为[N, 3]
            - total_data (np.ndarray): 所有三元组（正负合并），形状为[M, 3]，仅供需要时使用
            - meta (dict): 数据集的简要信息字典，包含样本数量、折数等元信息
    """
    # 加载并打乱正样本数据
    positive = load_positive(in_file, seed)                 # 正样本
    # 加载并打乱负样本全集
    negative_all = load_negative_all(neg_file, seed)        # 负样本全集
    # 从负样本全集中采样与正样本等量的负样本
    negative = sample_negative(negative_all, positive.shape[0])  # 与正样本等量采样
    # 为正负样本附加标签列（1表示正样本，0表示负样本）
    pos_l, neg_l, neg_all_l = attach_labels(positive, negative, negative_all)  # 附加标签
    # 进行K折交叉验证划分
    train_folds, test_folds = kfold_split_triples(pos_l, neg_l, k_fold=k_fold) # 五折划分

    # 合并所有带标签的正负样本
    total_data = np.vstack((pos_l, neg_l))
    # 构建数据集元信息
    meta = {
        "pos_count": int(pos_l.shape[0]),           # 正样本数量
        "neg_count": int(neg_l.shape[0]),           # 负样本数量（采样后的）
        "neg_all_count": int(neg_all_l.shape[0]),   # 负样本全集数量
        "folds": int(k_fold),                       # 折数
        "fold_size": int(pos_l.shape[0] // k_fold) if k_fold > 0 else int(pos_l.shape[0])  # 每折大小
    }
    return train_folds, test_folds, total_data, meta

# =================================================
# 多图对抗扰动内核
# =================================================
def adversarial_step_multi(
    X_list: List[torch.Tensor],
    loss_fn: Callable[[List[torch.Tensor]], torch.Tensor],
    cfg: Any
) -> List[torch.Tensor]:
    """
    为多个输入张量生成对抗扰动，实现多图对抗训练
    
    该函数通过对输入张量添加精心设计的扰动来生成对抗样本，可用于提高模型的鲁棒性。
    支持多种范数约束、共享/独立预算等多种配置选项。
    
    Args:
        X_list (List[torch.Tensor]): 输入张量列表，每个张量代表一个图的节点特征
        loss_fn (Callable[[List[torch.Tensor]], torch.Tensor]): 损失函数，接受扰动后的张量列表并返回标量损失值
        cfg (Any): 配置对象，包含各种对抗训练参数
        
    Returns:
        List[torch.Tensor]: 添加对抗扰动后的张量列表，形状与输入X_list相同
    """
    # 获取配置参数，设置默认值
    mode = getattr(cfg, "adv_mode", "none")
    if mode == "none":
        return X_list  # 未开启对抗

    # 对抗攻击参数配置
    # norm: 对抗攻击的范数类型，默认为 "linf" (L∞范数)
    # eps: 对抗扰动的最大幅度限制，默认为 0.01
    # alpha: 对抗攻击的步长，默认为 0.005
    # steps: 对抗攻击的迭代步数，默认为 0
    # rand_init: 是否随机初始化扰动，默认为 False
    # project: 是否将扰动投影到约束球内，默认为 True
    # agg: 梯度聚合方式，默认为 "mean"
    # budget: 对抗预算分配方式，默认为 "shared"
    # use_amp: 是否使用自动混合精度训练，默认为 False
    # cmin: 对抗样本特征值下界，默认为负无穷
    # cmax: 对抗样本特征值上界，默认为正无穷
    norm = getattr(cfg, "adv_norm", "linf")
    eps = float(getattr(cfg, "adv_eps", 0.01) or 0.0)
    alpha = float(getattr(cfg, "adv_alpha", 0.005) or 0.0)
    steps = int(getattr(cfg, "adv_steps", 0) or 0)
    rand_init = bool(getattr(cfg, "adv_rand_init", False))
    project = bool(getattr(cfg, "adv_project", True))
    agg = str(getattr(cfg, "adv_agg", "mean"))
    budget = str(getattr(cfg, "adv_budget", "shared"))
    use_amp = bool(getattr(cfg, "adv_use_amp", False))
    cmin = float(getattr(cfg, "adv_clip_min", float("-inf")))
    cmax = float(getattr(cfg, "adv_clip_max", float("inf")))

    # 确定设备
    device = X_list[0].device if len(X_list) > 0 else torch.device("cuda")

    # 初始化 delta（共享/独立）
    deltas: List[torch.Tensor] = []
    for X in X_list:
        if rand_init:
            deltas.append(maybe_rand_init_like(X, norm, eps).to(device))
        else:
            deltas.append(torch.zeros_like(X, device=device))

    # 单步退化为 FGSM（steps<=1）
    iters = max(1, steps)

    # AMP autocast 上下文（仅前向与 loss 计算）
    amp_ctx = torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else nullcontext

    # 迭代更新扰动
    for _ in range(iters):
        # 设置梯度追踪
        for d in deltas:
            d.requires_grad_(True)

        # 前向传播计算损失
        with amp_ctx():
            X_perturbed = [clamp_features(X.clone() + d, cmin, cmax) for X, d in zip(X_list, deltas)]
            loss = loss_fn(X_perturbed)

        # 反向传播获取梯度
        grads = torch.autograd.grad(loss, deltas, retain_graph=False, create_graph=False, allow_unused=False)

        # 更新 delta
        new_deltas: List[torch.Tensor] = []
        if budget == "shared":
            # 共享预算：对各图梯度范数做尺度对齐，使步长一致
            norms = [g.detach().view(g.size(0), -1).norm(p=2, dim=1).mean() for g in grads]
            norms = [(n + 1e-12) for n in norms]
            avg_norm = torch.stack([n if isinstance(n, torch.Tensor) else torch.tensor(float(n), device=device) for n in norms]).mean()
            scales = [(avg_norm / n) for n in norms]
            for (d, g, sc) in zip(deltas, grads, scales):
                upd = g * sc
                d_new = step_update(d, upd, norm, alpha)
                if project:
                    d_new = project_to_ball(d_new, norm, eps)
                new_deltas.append(d_new.detach())
        else:
            # 独立预算：每图独立按自身梯度更新并投影
            for d, g in zip(deltas, grads):
                d_new = step_update(d, g, norm, alpha)
                if project:
                    d_new = project_to_ball(d_new, norm, eps)
                new_deltas.append(d_new.detach())

        # 更新扰动并关闭梯度追踪
        deltas = new_deltas
        for d in deltas:
            d.requires_grad_(False)

        # 如果是FGSM（单步），则跳出循环
        if steps <= 1:
            break

    # 生成最终的对抗样本
    X_adv_list = [clamp_features(X + d, cmin, cmax).detach() for X, d in zip(X_list, deltas)]
    return X_adv_list

# =================================================
# 统一导出
# =================================================
__all__ = [
    # 通用/增强
    "reset_parameters", "AvgReadout",
    "random_permute_features", "add_noise", "attribute_mask", "noise_then_mask", "apply_augmentation",
    # MoCo/BYOL/融合/编码器/模型
    "MoCoV2MultiView", "BYOLMultiView", "BYOLLoss", "compute_byol_loss", "ModelFactory",
    "GraphTransformerStyleFusion", "FusionDecoder", "GATGTSerial", "EM",
    # 数据标注/三元组
    "load_positive", "load_negative_all", "sample_negative", "attach_labels", "kfold_split_triples", "build_triples",
    # 对抗扰动
    "adversarial_step_multi",
]