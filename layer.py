from typing import List, Optional, Any, Callable, Tuple
from contextlib import nullcontext
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

# =================================================
# EM 主模型（组合：编码器 + 融合 + MoCo）
# =================================================
class EM(nn.Module):
    """
    EM主模型类，组合了编码器、融合解码器和MoCo多视图对比学习模块
    
    该模型使用GAT-GT串行编码器进行节点表示学习，通过图变换器风格的融合解码器进行实体关系预测，
    并采用MoCo多视图对比学习框架进行自监督预训练。
    
    模型主要组件：
    - 编码器：GATGTSerial，先GATConv再TransformerConv的串联编码器
    - 融合解码器：FusionDecoder，使用GraphTransformerStyleFusion策略融合实体表示
    - 对比学习：MoCoV2MultiView，多视图动量对比学习框架
    - 对抗分支：用于节点级对抗训练的线性头
    
    Attributes:
        encoder (GATGTSerial): 图神经网络编码器
        read (AvgReadout): 图级读出函数
        mlp1 (nn.Linear): 图级表示的MLP变换层
        sigm (nn.Sigmoid): Sigmoid激活函数
        moco (MoCoV2MultiView): 多视图MoCo对比学习模块
        fusion (FusionDecoder): 融合解码器
        adv_head (nn.Linear): 对抗训练分支
        dropout (float): Dropout概率
        aug_list (List[str]): 数据增强方法列表
        noise_std (float): 噪声增强标准差
        mask_rate (float): 掩码增强比例
        base_seed (int): 增强随机种子基础值
    """
    def __init__(self, feature: int, hidden1: int, hidden2: int, decoder1: int, dropout: float):
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

        # MoCo（多视图）
        proj_dim = int(getattr(args, "proj_dim", hidden2) or hidden2)
        num_views = int(getattr(args, "num_views", 3) or 3)
        self.moco = MoCoV2MultiView(
            base_dim=hidden2,
            proj_dim=proj_dim,
            num_views=max(1, num_views),
            K=int(getattr(args, "moco_queue", 4096)),
            m=float(getattr(args, "moco_momentum", 0.999)),
            T=float(getattr(args, "moco_t", 0.2)),
            queue_warmup_steps=int(getattr(args, "queue_warmup_steps", 0)),
            debug=bool(getattr(args, "moco_debug", False)),
        )

        # 融合解码器（gt_fusion）
        heads = int(getattr(args, "fusion_heads", 4) or 4)
        self.fusion = FusionDecoder(hidden_dim=hidden2, decoder1_dim=decoder1, heads=heads, dropout=dropout)

        # 对抗分支
        self.adv_head = nn.Linear(hidden2, hidden2)

        # 训练态超参
        self.dropout = dropout

        # 增强设定
        self.aug_list = ["random_permute_features", "attribute_mask", "noise_then_mask"]
        self.noise_std = float(getattr(args, "noise_std", 0.01) or 0.01)
        self.mask_rate = float(getattr(args, "mask_rate", 0.1) or 0.1)
        self.base_seed = int(getattr(args, "augment_seed", getattr(args, "seed", 0)) or 0)

    def forward(self, data_o, data_a, idx):
        """
        前向传播函数
        
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

        # 多视图 MoCo：第0视图用损图，其余来自原图的增强
        num_views = int(getattr(args, "num_views", 3) or 3)
        k_embeds: List[torch.Tensor] = [x2_o_a]
        for vid in range(1, max(1, num_views)):
            seed_v = self.base_seed + vid
            aug_name = self.aug_list[(vid - 1) % len(self.aug_list)]
            x_aug = apply_augmentation(
                aug_name, x_o, noise_std=self.noise_std, mask_rate=self.mask_rate, seed=seed_v
            )
            x2_aug = self.encoder.encode(x_aug, edge_index)
            k_embeds.append(x2_aug)
        logits_list, targets_list = self.moco(x2_o, k_embeds)
        cla_os, cla_os_a = logits_list[0], targets_list[0]

        # 实体抽取（保持与旧逻辑兼容）
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

        # 融合解码（gt_fusion）
        log, log1 = self.fusion(entity1, entity2)

        # 对抗 logits（沿特征求和）
        sc_1 = self.adv_head(x2_o).sum(1).unsqueeze(0)
        sc_2 = self.adv_head(x2_o_a).sum(1).unsqueeze(0)
        logits_adv = torch.cat((sc_1, sc_2), 1)

        return log, cla_os, cla_os_a, x2_o, logits_adv, log1

# =================================================
# 数据标注/三元组构建（原 label_annotation.py）
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
            X_perturbed = [clamp_features(X + d, cmin, cmax) for X, d in zip(X_list, deltas)]
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
    # MoCo/融合/编码器/模型
    "MoCoV2MultiView", "GraphTransformerStyleFusion", "FusionDecoder", "GATGTSerial", "EM",
    # 数据标注/三元组
    "load_positive", "load_negative_all", "sample_negative", "attach_labels", "kfold_split_triples", "build_triples",
    # 对抗扰动
    "adversarial_step_multi",
]