from typing import List, Optional, Any, Callable, Tuple
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv
from parms_setting import settings
from utils import (
    em_path as _p,  # 路径解析
    maybe_rand_init_like as _maybe_rand_init_like,  # 对抗初始化
    clamp_features as _clamp,  # 特征裁剪
    step_update as _step_update,  # 梯度更新
    project_to_ball as _project,  # 范数投影
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
# 通用工具与轻量增强（已迁移至 utils.py 从此处导入）
# =================================================

# =================================================
# 编码器：gat_gt_serial（底层组件）
# =================================================
class GATGTSerial(nn.Module):
    # 先 GATConv 再 TransformerConv 的串联编码器
    def __init__(self, in_dim: int, hidden1: int, hidden2: int, dropout: float, gat_heads: int = 4):
        super().__init__()
        self.gat1 = GATConv(in_channels=in_dim, out_channels=hidden1, heads=gat_heads, concat=True, dropout=dropout)
        self.prelu_g1 = nn.PReLU(hidden1 * gat_heads)
        self.gt2 = TransformerConv(in_channels=hidden1 * gat_heads, out_channels=hidden2, heads=1, concat=False, dropout=dropout)
        self.prelu_t2 = nn.PReLU(hidden2)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 节点级编码：GAT -> Dropout -> Transformer -> PReLU
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
    """
    def __init__(self, hidden_dim: int, heads: int = 4, dropout: float = 0.1):
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
        # 将两个实体表示打包为长度为2的序列，进行自注意力与前馈，再展平拼接
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
    融合输出 -> 4H，再接 decoder1 -> 1，保持与训练兼容（返回 log, log1）
    """
    def __init__(self, hidden_dim: int, decoder1_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.strategy = GraphTransformerStyleFusion(hidden_dim, heads=heads, dropout=dropout)
        self.proj4h: nn.Module = nn.Linear(2 * hidden_dim, 4 * hidden_dim)  # 将[2H]映射到[4H]
        self.fc1 = nn.Linear(4 * hidden_dim, decoder1_dim)
        self.fc2 = nn.Linear(decoder1_dim, 1)

    def forward(self, e1: torch.Tensor, e2: torch.Tensor):
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
    多视图 MoCo v2：
    - 共享一个 q 投影头
    - 每个视图独立 k 投影头与队列
    - 返回每个视图的 (logits, targets)
    """
    def __init__(self, base_dim: int, proj_dim: int, num_views: int, K: int = 4096, m: float = 0.999, T: float = 0.2, queue_warmup_steps: int = 0, debug: bool = False):
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
        # 动量更新：将 k 编码器向 q 编码器移动
        for k_proj in self.k_projs:
            for param_q, param_k in zip(self.q_proj.parameters(), k_proj.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor, view_idx: int):
        # 出队入队：环形队列更新
        keys = keys.detach()
        batch_size = keys.shape[0]
        if batch_size <= 0:
            return
        K = self.K
        queue = getattr(self, f"queue_{view_idx}")
        queue_ptr = getattr(self, f"queue_ptr_{view_idx}")
        ptr = int(queue_ptr.item())
        if ptr + batch_size <= K:
            queue[:, ptr:ptr + batch_size] = keys.t()
            ptr = (ptr + batch_size) % K
        else:
            first = K - ptr
            second = batch_size - first
            if first > 0:
                queue[:, ptr:] = keys[:first].t()
            if second > 0:
                queue[:, :second] = keys[first:].t()
            ptr = second % K
        queue_ptr[0] = ptr

    def forward(self, q_embed: torch.Tensor, k_embeds: List[torch.Tensor]):
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
        warmup = self.global_step <= self.queue_warmup_steps

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
    EM: encoder = gat_gt_serial, fusion = gt_fusion, 对比 = MoCo 多视图
    返回：log, cla_os, cla_os_a, x2_o, logits_adv, log1
    """
    def __init__(self, feature: int, hidden1: int, hidden2: int, decoder1: int, dropout: float):
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
        ✅ 修复4：明确返回值顺序与类型注释
        返回：(log, cla_os, cla_os_a, x2_o, logits_adv, log1)
        - log: [B,1] 关联预测主任务输出
        - cla_os: Tensor MoCo 对比 logits（第0视图）
        - cla_os_a: Tensor MoCo 对比 targets（第0视图）
        - x2_o: [N,H] 原图节点表示（共享编码器输出）
        - logits_adv: [1,2N] 节点级对抗二分类 logits
        - log1: [B,decoder1] 融合解码器中间层输出
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
    读取并打乱正样本（已知关联），返回数组 shape=(N, 2)
    与 CSGLMD-main/data_preprocess.py 保持一致：保留全部样本，使用随机种子打乱
    """
    positive = np.loadtxt(_p(in_file), dtype=np.int64)
    link_size = int(positive.shape[0])  # 保留全部
    rng = np.random.default_rng(int(seed))  # 使用局部生成器，避免污染全局随机状态
    idx = rng.permutation(positive.shape[0])
    positive = positive[idx]
    positive = positive[:link_size]
    return positive

def load_negative_all(neg_file: str, seed: int):
    """
    读取并打乱负样本全集（未知关联），返回数组 shape=(M, 2)
    """
    negative_all = np.loadtxt(_p(neg_file), dtype=np.int64)
    rng = np.random.default_rng(int(seed))  # 使用局部生成器，避免污染全局随机状态
    idx = rng.permutation(negative_all.shape[0])
    negative_all = negative_all[idx]
    return negative_all

def sample_negative(negative_all: np.ndarray, pos_count: int):
    """
    采样与正样本等量的负样本（与参考实现完全一致）
    """
    if negative_all.shape[0] < pos_count:
        raise ValueError(f"负样本全集数量不足：需要 {pos_count}，实际 {negative_all.shape[0]}")
    negative = np.asarray(negative_all[:pos_count])
    return negative

def attach_labels(positive: np.ndarray, negative: np.ndarray, negative_all: np.ndarray):
    """
    为正/负样本分别附加标签列，输出：
    - positive_labeled: [i, j, 1]
    - negative_labeled: [i, j, 0]（采样得到，用于训练/测试）
    - negative_all_labeled: [i, j, 0]（全集，供需要时参考）
    """
    pos_lbl = np.ones(positive.shape[0], dtype=np.int64).reshape(positive.shape[0], 1)
    neg_lbl = np.zeros(negative.shape[0], dtype=np.int64).reshape(negative.shape[0], 1)
    neg_all_lbl = np.zeros(negative_all.shape[0], dtype=np.int64).reshape(negative_all.shape[0], 1)

    positive_labeled = np.concatenate([positive, pos_lbl], axis=1)
    negative_labeled = np.concatenate([negative, neg_lbl], axis=1)
    negative_all_labeled = np.concatenate([negative_all, neg_all_lbl], axis=1)

    return positive_labeled, negative_labeled, negative_all_labeled

def kfold_split_triples(positive_labeled: np.ndarray,
                        negative_labeled: np.ndarray,
                        k_fold: int = 5):
    """
    五折交叉划分，与参考实现一致：
    - 按正样本数量均分折；每折取对应区间为测试，其余为训练
    - 负样本按相同索引区间进行划分
    返回 train_data_folds, test_data_folds（列表，每项为 (num_samples, 3)）
    """
    if k_fold <= 0:
        raise ValueError("k_fold 必须为正整数")
    if positive_labeled.shape[0] != negative_labeled.shape[0]:
        raise ValueError("正负样本数量必须一致以进行等量划分")

    pos_num = positive_labeled.shape[0]
    fold_size = pos_num // k_fold
    train_data_folds = []
    test_data_folds = []

    for fold in range(k_fold):
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < k_fold - 1 else pos_num

        # 划分阳性样本
        test_positive = positive_labeled[start_idx:end_idx]
        train_positive = np.vstack((positive_labeled[:start_idx], positive_labeled[end_idx:]))

        # 划分阴性样本
        test_negative = negative_labeled[start_idx:end_idx]
        train_negative = np.vstack((negative_labeled[:start_idx], negative_labeled[end_idx:]))

        # 构建训练集和测试集
        train_data = np.vstack((train_positive, train_negative))
        test_data = np.vstack((test_positive, test_negative))

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
    返回：
    - train_data_folds: list[np.ndarray], 每折训练三元组
    - test_data_folds: list[np.ndarray], 每折测试三元组
    - total_data: np.ndarray, 所有三元组（正负合并，仅供需要时使用）
    - meta: dict, 简要信息
    """
    positive = load_positive(in_file, seed)                 # 正样本
    negative_all = load_negative_all(neg_file, seed)        # 负样本全集
    negative = sample_negative(negative_all, positive.shape[0])  # 与正样本等量采样
    pos_l, neg_l, neg_all_l = attach_labels(positive, negative, negative_all)  # 附加标签
    train_folds, test_folds = kfold_split_triples(pos_l, neg_l, k_fold=k_fold) # 五折划分

    total_data = np.vstack((pos_l, neg_l))
    meta = {
        "pos_count": int(pos_l.shape[0]),
        "neg_count": int(neg_l.shape[0]),
        "neg_all_count": int(neg_all_l.shape[0]),
        "folds": int(k_fold),
        "fold_size": int(pos_l.shape[0] // k_fold) if k_fold > 0 else int(pos_l.shape[0])
    }
    return train_folds, test_folds, total_data, meta

# =================================================
# 多图对抗扰动内核（原 adversarial.py）
# =================================================
def adversarial_step_multi(
    X_list: List[torch.Tensor],
    loss_fn: Callable[[List[torch.Tensor]], torch.Tensor],
    cfg: Any
) -> List[torch.Tensor]:
    """
    批内多图对抗生成（PGD/FGSM）

    参数：
    - X_list: List[Tensor] 多图特征列表，每个形状如 [N_i, D]
    - loss_fn: 闭包，接收当前扰动后的 X_list，返回标量损失（不需要 create_graph）
    - cfg: 任意对象（通常是 args），需包含以下属性（均有默认）：
        adv_mode: str = 'none' | 'mgraph'
        adv_norm: str = 'linf' | 'l2'
        adv_eps: float = 0.01
        adv_alpha: float = 0.005
        adv_steps: int = 0
        adv_rand_init: bool = False
        adv_project: bool = True
        adv_agg: str = 'mean' | 'sum' | 'max'
        adv_budget: str = 'shared' | 'independent'
        adv_use_amp: bool = False
        adv_clip_min: float = -inf
        adv_clip_max: float = +inf

    返回：
    - X_adv_list: List[Tensor] 与 X_list 同结构的扰动后特征
    """
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

    device = X_list[0].device if len(X_list) > 0 else torch.device("cuda")

    # 初始化 delta（共享/独立）
    deltas: List[torch.Tensor] = []
    for X in X_list:
        if rand_init:
            deltas.append(_maybe_rand_init_like(X, norm, eps).to(device))
        else:
            deltas.append(torch.zeros_like(X, device=device))

    # 单步退化为 FGSM（steps<=1）
    iters = max(1, steps)

    # AMP autocast 上下文（仅前向与 loss 计算）
    amp_ctx = torch.cuda.amp.autocast if (use_amp and device.type == "cuda") else nullcontext

    for _ in range(iters):
        for d in deltas:
            d.requires_grad_(True)

        # 前向与损失
        with amp_ctx():
            X_perturbed = [_clamp(X + d, cmin, cmax) for X, d in zip(X_list, deltas)]
            loss = loss_fn(X_perturbed)

        # 反传得梯度
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
                d_new = _step_update(d, upd, norm, alpha)
                if project:
                    d_new = _project(d_new, norm, eps)
                new_deltas.append(d_new.detach())
        else:
            # 独立预算：每图独立按自身梯度更新并投影
            for d, g in zip(deltas, grads):
                d_new = _step_update(d, g, norm, alpha)
                if project:
                    d_new = _project(d_new, norm, eps)
                new_deltas.append(d_new.detach())

        deltas = new_deltas
        for d in deltas:
            d.requires_grad_(False)

        if steps <= 1:
            break

    X_adv_list = [_clamp(X + d, cmin, cmax).detach() for X, d in zip(X_list, deltas)]
    return X_adv_list

# =================================================
# 统一导出（包含原 layer 内的核心符号与新并入的工具）
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