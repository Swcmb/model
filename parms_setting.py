import argparse  # 命令行参数解析


def settings():
    """
    构建并解析实验参数，返回包含所有设置的命名空间对象。
    仅调整组织顺序与中文注释，不改变任何参数名、默认值、choices 或逻辑。
    """
    # 创建解析器
    parser = argparse.ArgumentParser()

    # ==================== 公共参数 ====================
    parser.add_argument('--seed', type=int, default=0,
                        help='随机种子，默认 0。')
    # --file 作为 --in_file 的别名
    parser.add_argument('--file', dest='in_file', default="dataset1/LDA.edgelist",
                        help='正样本文件（--in_file 的别名）。')
    parser.add_argument('--neg_sample', default="dataset1/non_LDA.edgelist",
                        help='未知关联（负样本）文件路径。')
    parser.add_argument('--validation_type', default="5_cv1", choices=['5_cv1', '5_cv2', '5-cv1', '5-cv2'],
                        help='交叉验证类型，默认 5_cv1。')
    parser.add_argument('--task_type', default="LDA", choices=['LDA', 'MDA', 'LMI'],
                        help='任务类型：LDA/MDA/LMI，默认 LDA。')

    # ==================== 特征构建与增强 ====================
    parser.add_argument('--feature_type', type=str, default='normal',
                        choices=['one_hot', 'uniform', 'normal', 'position'],
                        help='初始节点特征类型，默认 normal。')
    parser.add_argument('--noise_std', type=float, default=0.01,
                        help='add_noise / noise_then_mask 的高斯噪声标准差，默认 0.01。')
    parser.add_argument('--mask_rate', type=float, default=0.1,
                        help='attribute_mask / noise_then_mask 的列掩蔽比例，默认 0.1。')
    parser.add_argument('--augment_seed', type=int, default=None,
                        help='增强随机种子；None 时使用 seed+fold。')
    parser.add_argument('--augment_mode', type=str, default='static',
                        choices=['static', 'online'],
                        help='增强模式：static（按折离线）/ online（训练时在线），默认 static。')

    # ==================== 训练设置 ====================
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='学习率，默认 5e-4。')
    # --learning_rate 作为 --lr 的别名
    parser.add_argument('--learning_rate', dest='lr', type=float,
                        help='--lr 的别名。')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout 比例，默认 0.1。')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='权重衰减（L2 正则），默认 5e-4。')
    parser.add_argument('--batch', type=int, default=25,
                        help='批大小，默认 25。')
    parser.add_argument('--epochs', type=int, default=1,
                        help='训练轮数，默认 1。')

    # 多任务损失权重
    parser.add_argument('--loss_ratio1', type=float, default=1,
                        help='任务1损失权重，默认 1。')
    parser.add_argument('--loss_ratio2', type=float, default=0.5,
                        help='任务2损失权重，默认 0.5。')
    parser.add_argument('--loss_ratio3', type=float, default=0.5,
                        help='任务3损失权重，默认 0.5。')

    # ==================== 模型结构参数 ====================
    parser.add_argument('--dimensions', type=int, default=256,
                        help='初始特征维度 d，默认 256（LDA/MDA）。')
    # --embed_dim 作为 --dimensions 的别名
    parser.add_argument('--embed_dim', dest='dimensions', type=int,
                        help='--dimensions 的别名。')
    parser.add_argument('--hidden1', type=int, default=128,
                        help='编码器第 1 层隐藏维度，默认 d/2。')
    parser.add_argument('--hidden2', type=int, default=64,
                        help='编码器第 2 层隐藏维度，默认 d/4。')
    parser.add_argument('--decoder1', type=int, default=512,
                        help='解码器第 1 层隐藏维度，默认 512。')
    # 注意力头
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='GAT 编码器的注意力头数，默认 4。')
    parser.add_argument('--gt_heads', type=int, default=4,
                        help='Graph Transformer 编码器的注意力头数，默认 4。')
    # 融合策略
    parser.add_argument('--fusion_heads', type=int, default=4,
                        help='对偶融合的多头注意力头数，默认 4。')

    # ==================== MoCo v2 / 对比学习 ====================
    parser.add_argument('--moco_queue', type=int, default=4096,
                        help='MoCo 队列长度，默认 4096。')
    parser.add_argument('--moco_momentum', type=float, default=0.999,
                        help='MoCo 动量 m，默认 0.999。')
    parser.add_argument('--moco_t', type=float, default=0.2,
                        help='MoCo 温度 T，默认 0.2。')
    parser.add_argument('--proj_dim', type=int, default=None,
                        help='投影维度，默认随 hidden2。')
    parser.add_argument('--queue_warmup_steps', type=int, default=0,
                        help='队列预热步数（预热期间仅用 batch 内负样本）。')
    parser.add_argument('--moco_debug', type=int, default=0,
                        help='轻量级 MoCo 调试日志开关（0/1）。')

    # ==================== CPU 并行与数据加载 ====================
    parser.add_argument('--threads', type=int, default=32,
                        help='后端线程上限；-1 自动探测（上限 32）。')
    parser.add_argument('--num_workers', type=int, default=-1,
                        help='DataLoader workers；-1 自动探测，上限 32（默认 min(8, threads)）。')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='DataLoader 预取因子（仅在 num_workers>0 时有效）。')
    parser.add_argument('--chunk_size', type=int, default=0,
                        help='CPU 任务通用切片大小；0 自动（默认 20000）。')

    # ==================== 新增需求支持参数 ====================
    parser.add_argument('--similarity_threshold', type=float, default=0.5,
                        help='图构建中的相似度阈值，默认 0.5。')

    # 别名权重：alpha=监督、beta=对比、gamma=节点对抗
    parser.add_argument('--alpha', dest='loss_ratio1', type=float, default=1.0,
                        help='监督任务权重（BCE），--loss_ratio1 别名。')
    parser.add_argument('--beta', dest='loss_ratio2', type=float, default=0.5,
                        help='对比任务权重（InfoNCE/CE），--loss_ratio2 别名。')
    parser.add_argument('--gamma', dest='loss_ratio3', type=float, default=0.5,
                        help='节点对抗任务权重（BCEWithLogits），--loss_ratio3 别名。')

    # ==================== 数据保存与折数 ====================
    parser.add_argument('--save_datasets', type=lambda x: str(x).lower() == 'true', default=False,
                        help='是否保存构建的数据集（true/false），默认 false。')
    parser.add_argument('--save_format', type=str, default='npy', choices=['npy', 'txt'],
                        help='数据保存格式，默认 npy。')
    parser.add_argument('--save_dir_prefix', type=str, default='result/data',
                        help='保存目录前缀，相对 EM/ 路径，默认 result/data。')

    # ==================== 运行控制 ====================
    parser.add_argument('--run_name', type=str, default=None,
                        help='运行名称，用于日志与结果目录前缀展示。')
    parser.add_argument('--shutdown', action='store_true',
                        help='仅 Linux：运行结束后关机。')

    # ==================== 对抗学习与 K 折重算/缓存 ====================
    # 对抗模式与核心超参
    parser.add_argument('--adv_mode', type=str, default='none',
                        choices=['none', 'mgraph'],
                        help='对抗模式：none | mgraph（批内多图对抗），默认 none。')
    parser.add_argument('--adv_norm', type=str, default='linf',
                        choices=['linf', 'l2'],
                        help='对抗范数：linf 或 l2，默认 linf。')
    parser.add_argument('--adv_eps', type=float, default=0.01,
                        help='对抗总预算 epsilon，默认 0.01。')
    parser.add_argument('--adv_alpha', type=float, default=0.005,
                        help='单步步长 alpha，默认 0.005。')
    parser.add_argument('--adv_steps', type=int, default=0,
                        help='PGD 步数；0/1 等价 FGSM，默认 0。')
    parser.add_argument('--adv_rand_init', type=lambda x: str(x).lower() == 'true', default=False,
                        help='是否随机初始化对抗增量（true/false），默认 false。')
    parser.add_argument('--adv_project', type=lambda x: str(x).lower() == 'true', default=True,
                        help='每步后是否投影回 epsilon-ball（true/false），默认 true。')
    # 批内多图策略
    parser.add_argument('--adv_agg', type=str, default='mean',
                        choices=['mean', 'sum', 'max'],
                        help='批内多图梯度聚合：mean/sum/max，默认 mean。')
    parser.add_argument('--adv_budget', type=str, default='shared',
                        choices=['shared', 'independent'],
                        help='跨图预算方式：shared/independent，默认 shared。')
    # 训练细节
    parser.add_argument('--adv_use_amp', type=lambda x: str(x).lower() == 'true', default=False,
                        help='对抗生成是否启用 AMP（true/false），默认 false。')
    parser.add_argument('--adv_on_moco', type=lambda x: str(x).lower() == 'true', default=False,
                        help='对抗输入是否送入 MoCo/对比分支（true/false），默认 false。')
    parser.add_argument('--adv_seed', type=int, default=None,
                        help='对抗生成随机种子；None 使用 seed+fold+iter。')
    parser.add_argument('--adv_clip_min', type=float, default=float("-inf"),
                        help='扰动后的最小裁剪值，默认 -inf。')
    parser.add_argument('--adv_clip_max', type=float, default=float("inf"),
                        help='扰动后的最大裁剪值，默认 +inf。')
    # K 折重算与缓存
    parser.add_argument('--kfold_recompute', type=lambda x: str(x).lower() == 'true', default=True,
                        help='按折仅用训练集重算预处理/相似度/EM（true/false），默认 true。')
    parser.add_argument('--kfold_cache', type=lambda x: str(x).lower() == 'true', default=False,
                        help='按 {fold, adv_hash, epoch, iter} 可选缓存对抗样本（true/false），默认 false。')

    # 解析
    args = parser.parse_args()

    # ==================== 解析后规范化与兜底 ====================
    # 统一 validation_type 格式：5-cvX -> 5_cvX
    if args.validation_type == '5-cv1':
        args.validation_type = '5_cv1'
    elif args.validation_type == '5-cv2':
        args.validation_type = '5_cv2'

    # MoCo proj_dim 兜底：None 或非法值时跟随 hidden2
    try:
        pd = getattr(args, "proj_dim", None)
        if pd is None or int(pd) <= 0:
            args.proj_dim = args.hidden2
    except Exception:
        args.proj_dim = args.hidden2

    return args