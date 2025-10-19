import platform               # OS 判定（用于控制并行策略）
import hashlib                # 轻量哈希（折级统计用）
from datetime import datetime # 时间戳与日志记录
import numpy as np            # 数值计算
import torch                  # 深度学习张量与设备管理
import scipy.sparse as sp     # 稀疏矩阵运算
from torch.utils.data import Dataset, DataLoader  # 数据集与加载器
from torch_geometric.data import Data             # 图数据封装（PyTorch Geometric）
from utils import *                         # 通用工具（含 BASE_DIR、图构建、归一化等）
from utils import em_path as _p             # 统一路径解析（简写）
from layer import load_positive, load_negative_all, sample_negative, attach_labels, apply_augmentation# 样本构建与特征增强
from calculating_similarity import calculate_GaussianKernel_sim, getRNA_functional_sim, RNA_fusion_sim, dis_fusion_sim# 相似度计算
from log_output_manager import get_logger, save_cv_datasets, save_fold_stats_json# 日志与数据保存


# ===== 说明：本模块默认使用五折交叉验证 =====


# =================================================
# 数据集定义
# =================================================
class Data_class(Dataset):
    """三元组数据集：返回 (label, (entity1, entity2))"""
    def __init__(self, triple):
        # triple 期望形状为 [N, 3]，列分别为 entity1, entity2, label
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.label = triple[:, 2]

    def __len__(self):
        # 数据集大小
        return len(self.label)

    def __getitem__(self, index):
        # 单样本：返回 (label, (entity1, entity2))
        return self.label[index], (self.entity1[index], self.entity2[index])


# =================================================
# 折数据访问辅助
# =================================================
def get_fold_data(data_o, data_a, train_loaders, test_loaders, fold_index):
    """获取指定折的数据加载器与特征视图"""
    if fold_index >= len(train_loaders) or fold_index < 0:
        raise ValueError(f"Fold index {fold_index} is out of range. Available folds: 0-{len(train_loaders)-1}")
    return data_o, data_a, train_loaders[fold_index], test_loaders[fold_index]


# =================================================
# 主流程：读取数据并构建五折交叉验证
# =================================================
def load_data(args, k_fold=5):
    """从路径读取数据，转换为 k 折交叉验证加载器，返回特征与邻接"""
    # 日志：运行配置
    _logger = get_logger()
    _logger.info('Loading {0} seed{1} dataset...'.format(args.in_file, args.seed))
    _logger.info(f"Selected cross_validation type: {args.validation_type}")
    _logger.info(f"Selected task_type: {args.task_type}")
    _logger.info(f"Selected feature_type: {args.feature_type}")
    _logger.info(f"Selected embed_dim: {getattr(args, 'embed_dim', 'N/A')}")
    _logger.info(f"Selected learning_rate: {getattr(args, 'learning_rate', 'N/A')}")
    _logger.info(f"Selected epochs: {getattr(args, 'epochs', 'N/A')}")

    # 读取正样本与负样本全集
    positive = load_positive(args.in_file, args.seed)  # shape=(P,2)
    negative_all = load_negative_all(args.neg_sample, args.seed)  # shape=(N,2)

    # 为正样本附加标签（1）
    pos_lbl = np.ones(positive.shape[0], dtype=np.int64).reshape(positive.shape[0], 1)
    positive_labeled = np.concatenate([positive, pos_lbl], axis=1)

    # 容器
    train_data_folds = []
    test_data_folds = []
    train_loaders = []
    test_loaders = []

    # 两种折分方案：5_cv2 与 默认 5_cv1
    if args.validation_type == '5_cv2':
        # 5-cv2：
        # - 正样本分 5 折；训练用 4 折正样本 + 等量随机负样本；测试用 1 折正样本 + 全部负样本
        fold_size = positive.shape[0] // 5

        # 全负样本附加标签（测试集使用全部负样本）
        neg_all_lbl = np.zeros(negative_all.shape[0], dtype=np.int64).reshape(negative_all.shape[0], 1)
        negative_all_labeled = np.concatenate([negative_all, neg_all_lbl], axis=1)

        for fold in range(5):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < 4 else positive.shape[0]

            test_positive = positive_labeled[start_idx:end_idx]
            train_positive = np.vstack((positive_labeled[:start_idx], positive_labeled[end_idx:]))

            # 训练负样本：等量随机采样（局部生成器，避免污染全局）
            rng = np.random.default_rng(int(args.seed) + fold)
            neg_shuffled = negative_all.copy()
            idx = rng.permutation(neg_shuffled.shape[0])
            neg_shuffled = neg_shuffled[idx]
            train_neg_sampled = np.asarray(neg_shuffled[:train_positive.shape[0]])
            train_neg_lbl = np.zeros(train_neg_sampled.shape[0], dtype=np.int64).reshape(train_neg_sampled.shape[0], 1)
            train_negative = np.concatenate([train_neg_sampled, train_neg_lbl], axis=1)

            # 测试负样本：全部负样本
            test_negative = negative_all_labeled

            # 拼接训练集/测试集
            train_data = np.vstack((train_positive, train_negative))
            test_data = np.vstack((test_positive, test_negative))

            train_data_folds.append(train_data)
            test_data_folds.append(test_data)

        total_data = np.vstack((positive_labeled, negative_all_labeled))

    else:
        # 默认 5_cv1：
        # - 负样本采样为与正样本等量
        # - 正负样本按同一索引区间切分为 5 折
        neg_sampled = sample_negative(negative_all, positive.shape[0])
        neg_lbl = np.zeros(neg_sampled.shape[0], dtype=np.int64).reshape(neg_sampled.shape[0], 1)
        negative_labeled = np.concatenate([neg_sampled, neg_lbl], axis=1)

        fold_size = positive.shape[0] // 5

        for fold in range(5):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < 4 else positive.shape[0]

            # 阳性划分
            test_positive = positive_labeled[start_idx:end_idx]
            train_positive = np.vstack((positive_labeled[:start_idx], positive_labeled[end_idx:]))

            # 阴性划分（同样索引区间）
            test_negative = negative_labeled[start_idx:end_idx]
            train_negative = np.vstack((negative_labeled[:start_idx], negative_labeled[end_idx:]))

            # 拼接训练集/测试集
            train_data = np.vstack((train_positive, train_negative))
            test_data = np.vstack((test_positive, test_negative))

            train_data_folds.append(train_data)
            test_data_folds.append(test_data)

        total_data = np.vstack((positive_labeled, negative_labeled))

    # （可选）保存折数据，由 log_output_manager 统一实现
    if getattr(args, 'save_datasets', True):
        save_cv_datasets(args, total_data, train_data_folds, test_data_folds, BASE_DIR)

    _logger.info('Selected task type...')

    # 每折输出容器
    data_o_folds = []
    data_a_folds = []
    fold_stats = []  # 收集每折统计与哈希

    # 疾病语义相似度（固定来源文件）
    dis_sem_sim = np.loadtxt(_p("dataset1/dis_sem_sim.txt"))

    # ✅ 修复1：移除硬编码第0折选择，改为每折独立构建图

    def mask_pairs(mat, pairs):
        """将测试集关联位置置 0（临时掩码，向量化实现）"""
        if pairs is None:
            return
        try:
            if len(pairs) == 0:
                return
        except TypeError:
            return
        p = np.asarray(pairs, dtype=int)
        if p.ndim != 2 or p.shape[1] != 2:
            return
        r = p[:, 0]
        c = p[:, 1]
        mask = (r >= 0) & (r < mat.shape[0]) & (c >= 0) & (c < mat.shape[1])
        if np.any(mask):
            mat[r[mask], c[mask]] = 0

    for fold in range(5):
        train_data = train_data_folds[fold]
        test_data = test_data_folds[fold]
        train_positive = train_data[train_data[:, 2] == 1]
        test_positive = test_data[test_data[:, 2] == 1]

        # 基于训练集构建 inter-layer，并对测试位置掩码
        if args.task_type == 'LDA':
            # lncRNA-disease
            l_d = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                                shape=(240, 405), dtype=np.float32).toarray()
            mask_pairs(l_d, test_positive[:, :2].astype(int))

            # 其他关联来源原始数据
            m_d = np.loadtxt(_p("dataset1/mi_dis.txt"))
            m_l = np.loadtxt(_p("dataset1/lnc_mi.txt")).T

            # 训练集重算融合相似性
            lnc_gau_1 = calculate_GaussianKernel_sim(l_d)
            lnc_gau_2 = calculate_GaussianKernel_sim(m_l.T)
            lnc_fun = getRNA_functional_sim(RNAlen=l_d.shape[0], diSiNet=dis_sem_sim.copy(), rna_di=l_d.copy())
            l_sim = RNA_fusion_sim(lnc_gau_1, lnc_gau_2, lnc_fun)

            mi_gau_1 = calculate_GaussianKernel_sim(m_d)
            mi_gau_2 = calculate_GaussianKernel_sim(m_l)
            mi_fun = getRNA_functional_sim(RNAlen=m_d.shape[0], diSiNet=dis_sem_sim.copy(), rna_di=m_d.copy())
            m_sim = RNA_fusion_sim(mi_gau_1, mi_gau_2, mi_fun)

            dis_gau_1 = calculate_GaussianKernel_sim(l_d.T)
            dis_gau_2 = calculate_GaussianKernel_sim(m_d.T)
            d_sim = dis_fusion_sim(dis_gau_1, dis_gau_2, dis_sem_sim)

        elif args.task_type == 'MDA':
            # miRNA-disease
            m_d = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                                shape=(495, 405), dtype=np.float32).toarray()
            mask_pairs(m_d, test_positive[:, :2].astype(int))

            l_d = np.loadtxt(_p("dataset1/lnc_dis.txt"))
            m_l = np.loadtxt(_p("dataset1/lnc_mi.txt")).T

            lnc_gau_1 = calculate_GaussianKernel_sim(l_d)
            lnc_gau_2 = calculate_GaussianKernel_sim(m_l.T)
            lnc_fun = getRNA_functional_sim(RNAlen=l_d.shape[0], diSiNet=dis_sem_sim.copy(), rna_di=l_d.copy())
            l_sim = RNA_fusion_sim(lnc_gau_1, lnc_gau_2, lnc_fun)

            mi_gau_1 = calculate_GaussianKernel_sim(m_d)
            mi_gau_2 = calculate_GaussianKernel_sim(m_l)
            mi_fun = getRNA_functional_sim(RNAlen=m_d.shape[0], diSiNet=dis_sem_sim.copy(), rna_di=m_d.copy())
            m_sim = RNA_fusion_sim(mi_gau_1, mi_gau_2, mi_fun)

            dis_gau_1 = calculate_GaussianKernel_sim(l_d.T)
            dis_gau_2 = calculate_GaussianKernel_sim(m_d.T)
            d_sim = dis_fusion_sim(dis_gau_1, dis_gau_2, dis_sem_sim)

        elif args.task_type == 'LMI':
            # lncRNA-miRNA
            l_m = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                                shape=(240, 495), dtype=np.float32).toarray()
            # miRNA-lncRNA
            m_l = l_m.T
            # 掩码时索引反转
            mask_pairs(m_l, np.ascontiguousarray(test_positive[:, :2][:, ::-1]).astype(int))

            l_d = np.loadtxt(_p("dataset1/lnc_dis.txt"))
            m_d = np.loadtxt(_p("dataset1/mi_dis.txt"))

            lnc_gau_1 = calculate_GaussianKernel_sim(l_d)
            lnc_gau_2 = calculate_GaussianKernel_sim(m_l.T)
            lnc_fun = getRNA_functional_sim(RNAlen=l_d.shape[0], diSiNet=dis_sem_sim.copy(), rna_di=l_d.copy())
            l_sim = RNA_fusion_sim(lnc_gau_1, lnc_gau_2, lnc_fun)

            mi_gau_1 = calculate_GaussianKernel_sim(m_d)
            mi_gau_2 = calculate_GaussianKernel_sim(m_l)
            mi_fun = getRNA_functional_sim(RNAlen=m_d.shape[0], diSiNet=dis_sem_sim.copy(), rna_di=m_d.copy())
            m_sim = RNA_fusion_sim(mi_gau_1, mi_gau_2, mi_fun)

            dis_gau_1 = calculate_GaussianKernel_sim(l_d.T)
            dis_gau_2 = calculate_GaussianKernel_sim(m_d.T)
            d_sim = dis_fusion_sim(dis_gau_1, dis_gau_2, dis_sem_sim)

        else:
            raise ValueError(f"Unknown task_type: {args.task_type}")

        # 构建邻接并归一化
        adj = construct_graph(l_d, m_d, m_l, l_sim, m_sim, d_sim)
        adj = lalacians_norm(adj)

        # 边索引
        edges_o = adj.nonzero()
        edge_index_o = torch.tensor(np.vstack((edges_o[0], edges_o[1])), dtype=torch.long)

        # 特征构建
        if args.feature_type == 'one_hot':
            features = np.eye(adj.shape[0])
        elif args.feature_type == 'uniform':
            rng = np.random.default_rng(int(args.seed))
            features = rng.uniform(low=0, high=1, size=(adj.shape[0], args.dimensions))
        elif args.feature_type == 'normal':
            rng = np.random.default_rng(int(args.seed))
            features = rng.normal(loc=0, scale=1, size=(adj.shape[0], args.dimensions))
        elif args.feature_type == 'position':
            features = sp.coo_matrix(adj).todense()
        else:
            features = np.eye(adj.shape[0])

        features_o = normalize(features)
        if fold == 0:
            args.dimensions = features_o.shape[1]

        # 对抗/增强特征视图：根据 --augment 选择增强（默认 random_permute_features）
        aug_name = getattr(args, "augment", "random_permute_features")
        # 兼容多选增强：静态构图阶段仅取第一个增强名
        if isinstance(aug_name, (list, tuple)):
            aug_name = aug_name[0] if len(aug_name) > 0 else "random_permute_features"
        noise_std = float(getattr(args, "noise_std", 0.01) or 0.01)
        mask_rate = float(getattr(args, "mask_rate", 0.1) or 0.1)
        base_seed = getattr(args, "augment_seed", None)
        if base_seed is None:
            base_seed = int(getattr(args, "seed", 0)) + fold

        _aug_key = (aug_name or "").strip().lower()
        # 将特征放到 GPU/CPU（按 args.cuda）上，增强直接在 Tensor 上进行
        _device = torch.device("cuda") if getattr(args, "cuda", False) and torch.cuda.is_available() else torch.device("cpu")
        x_o = torch.tensor(features_o, dtype=torch.float, device=_device)
        if _aug_key in {"", "none", "null"}:
            # 无增强：直接引用原特征张量
            features_a = x_o
        else:
            features_a = apply_augmentation(
                aug_name,
                x_o,
                noise_std=noise_std,
                mask_rate=mask_rate,
                seed=base_seed
            )

        # 记录增强统计（全 Torch 计算，避免 numpy 回落）与折级统计
        try:
            _alog = get_logger("augment")
            masked_cols = int((features_a == 0).all(dim=0).sum().item())
            mean_o = float(x_o.mean().item())
            std_o = float(x_o.float().std(unbiased=False).item()) if x_o.numel() > 1 else 0.0
            mean_a = float(features_a.mean().item())
            std_a = float(features_a.float().std(unbiased=False).item()) if features_a.numel() > 1 else 0.0
            _shape = tuple(features_a.shape)
            _alog.info(f"[AUGMENT][fold={fold+1}] name={aug_name} noise_std={noise_std} mask_rate={mask_rate} seed={base_seed} masked_cols={masked_cols} shape={_shape} mean={mean_a:.4f} std={std_a:.4f}")
        except Exception:
            masked_cols = 0
            mean_o = std_o = mean_a = std_a = 0.0

        # 相似度与图的轻量哈希（跨折可比，不写出大矩阵）
        try:
            def _sha1_arr(arr: np.ndarray) -> str:
                h = hashlib.sha1()
                h.update(arr.tobytes())
                return h.hexdigest()[:16]
            hash_l = _sha1_arr(l_sim.astype(np.uint8)) if 'l_sim' in locals() else "-"
            hash_m = _sha1_arr(m_sim.astype(np.uint8)) if 'm_sim' in locals() else "-"
            hash_d = _sha1_arr(d_sim.astype(np.uint8)) if 'd_sim' in locals() else "-"
        except Exception:
            hash_l = hash_m = hash_d = "-"

        # 记录每折统计（训练/测试规模、mask 数、特征统计、增强配置、相似度哈希）
        try:
            fold_stats.append({
                "fold": fold + 1,
                "train_size": int(train_data.shape[0]),
                "test_size": int(test_data.shape[0]),
                "pos_train": int((train_data[:,2] == 1).sum()),
                "pos_test": int((test_data[:,2] == 1).sum()),
                "masked_cols_in_aug": int(masked_cols),
                "features_o": {"mean": mean_o, "std": std_o, "shape": list(features_o.shape)},
                "features_a": {"mean": mean_a, "std": std_a, "shape": list(features_a.shape)},
                "augment": {"name": str(aug_name), "noise_std": float(noise_std), "mask_rate": float(mask_rate), "seed": int(base_seed)},
                "similarity_hash": {"lnc": hash_l, "mi": hash_m, "dis": hash_d}
            })
        except Exception:
            pass

        # y_a：对抗视图的二分类标签（未使用占位，保持与下游兼容）
        y_a = torch.cat((torch.ones(adj.shape[0], 1), torch.zeros(adj.shape[0], 1)), dim=1).to(x_o.device)

        # 构造图数据对象（边索引放同设备，减少搬运）
        data_o = Data(x=x_o, edge_index=edge_index_o.to(x_o.device))
        data_a = Data(x=features_a, y=y_a)

        data_o_folds.append(data_o)
        data_a_folds.append(data_a)

    # 为所有折构建 DataLoader（并行策略由 autodl 决策）
    os_name = platform.system().lower()
    num_workers = decide_dataloader_workers(args)
    prefetch_factor = int(getattr(args, "prefetch_factor", 4) or 4)

    base_params = {'batch_size': args.batch, 'shuffle': True, 'drop_last': True}
    if num_workers > 0:
        base_params.update({
            'num_workers': num_workers,
            'persistent_workers': True,
            'pin_memory': False
        })
        if prefetch_factor and prefetch_factor > 0:
            base_params['prefetch_factor'] = prefetch_factor

    # 记录一次实际使用的 workers 策略
    _logger = get_logger()
    _logger.info(f"[DATALOADER] os={os_name} workers={num_workers} prefetch_factor={(base_params.get('prefetch_factor') if num_workers>0 else 0)}")

    train_loaders = []
    test_loaders = []
    for fold in range(5):
        training_set = Data_class(train_data_folds[fold])
        train_loaders.append(DataLoader(training_set, **base_params))
        test_set = Data_class(test_data_folds[fold])
        test_loaders.append(DataLoader(test_set, **base_params))

    # 写出折级统计（OUTPUT/result/metrics）
    save_fold_stats_json(fold_stats, BASE_DIR)

    _logger.info('Loading finished!')
    return data_o_folds, data_a_folds, train_loaders, test_loaders