import torch
import torch.nn as nn
import torch.nn.functional as F


# =================================================
# 融合模块（gt_fusion）
# =================================================
class GraphTransformerStyleFusion(nn.Module):
    """
    Graph Transformer 风格的两-token注意力 + 前馈；输出拼接为 [B, 2H]
    
    该模块将两个实体表示作为输入，通过自注意力机制和前馈网络进行特征融合，
    最终将两个实体的表示拼接为一个长度为2H的向量。
    
    Attributes:
        mha (nn.MultiheadAttention): 多头注意力机制层
        ffn (nn.Sequential): 前馈神经网络层
        norm1 (nn.LayerNorm): 第一层归一化层
        norm2 (nn.LayerNorm): 第二层归一化层
        dropout (nn.Dropout): Dropout层
    """
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1):
        """
        初始化GraphTransformerStyleFusion模块
        
        Args:
            hidden_dim (int): 隐藏层维度
            heads (int, optional): 注意力头的数量，默认为4
            dropout (float, optional): Dropout概率，默认为0.1
        """
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，对两个实体表示进行特征融合
        
        将两个实体表示打包为长度为2的序列，进行自注意力与前馈操作，再展平拼接
        
        Args:
            e1 (torch.Tensor): 第一个实体的表示，形状为 [B, H]
            e2 (torch.Tensor): 第二个实体的表示，形状为 [B, H]
            
        Returns:
            torch.Tensor: 融合后的表示，形状为 [B, 2H]
        """
        B, H = e1.size(0), e1.size(1)
        x = torch.stack([e1, e2], dim=1)          # [B,2,H]
        attn_out, _ = self.mha(x, x, x)           # [B,2,H]
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out)) # [B,2,H]
        x = x.reshape(B, 2 * H)                   # [B,2H]
        return x

class FusionDecoder(nn.Module):
    """
    融合解码器，用于将两个实体的表示融合后解码为二分类分数
    
    该解码器首先使用GraphTransformerStyleFusion策略将两个实体表示融合，
    然后通过两层全连接网络将融合后的特征映射到最终的二分类分数。
    
    Attributes:
        hidden_dim (int): 隐藏层维度
        strategy (GraphTransformerStyleFusion): 融合策略模块
        proj4h (nn.Linear): 将2H维度映射到4H维度的线性变换层
        fc1 (nn.Linear): 第一层全连接层
        fc2 (nn.Linear): 第二层全连接层，输出维度为1
    """
    def __init__(self, hidden_dim: int, decoder1_dim: int, heads: int = 4, dropout: float = 0.1):
        """
        初始化融合解码器
        
        Args:
            hidden_dim (int): 隐藏层维度
            decoder1_dim (int): 第一层解码器的输出维度
            heads (int, optional): 注意力头数，默认为4
            dropout (float, optional): Dropout概率，默认为0.1
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.strategy = GraphTransformerStyleFusion(hidden_dim, heads=heads, dropout=dropout)
        self.proj4h: nn.Module = nn.Linear(2 * hidden_dim, 4 * hidden_dim)  # 将[2H]映射到[4H]
        self.fc1 = nn.Linear(4 * hidden_dim, decoder1_dim)
        self.fc2 = nn.Linear(decoder1_dim, 1)

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
        feat2h = self.strategy(e1, e2)            # [B,2H]
        fused4h = self.proj4h(feat2h)             # [B,4H]
        log1 = F.relu(self.fc1(fused4h))          # [B,decoder1]
        log = self.fc2(log1)                      # [B,1]
        return log, log1
