import numpy as np  # 数值计算与数组/矩阵操作
import copy  # 用于创建对象副本（浅拷贝/深拷贝均可）


# ========= 预处理与核带宽 =========
def Preproces_Data(A, test_id):
    """
    将测试集中的阳性样本在关联矩阵 A 中置 0（不改动原始矩阵）
    
    参数:
        A: 关联矩阵
        test_id: 测试集索引矩阵，每行包含两个元素，分别表示关联矩阵中的行索引和列索引
        
    返回:
        copy_A: 修改后的关联矩阵副本，其中测试集中的阳性样本位置被置为0
    """
    copy_A = A / 1  # 创建 A 的副本，避免修改原数据
    for i in range(test_id.shape[0]):
        copy_A[int(test_id[i][0])][int(test_id[i][1])] = 0
    return copy_A


def calculate_kernel_bandwidth(A):
    """
    计算高斯核带宽参数（基于每行谱的 L2 范数平方的平均值的倒数）
    
    参数:
        A: 输入的关联矩阵，用于计算高斯核带宽参数
        
    返回:
        lambd: 高斯核带宽参数，为每行向量L2范数平方平均值的倒数
    """
    IP_0 = 0
    # 累加所有行向量的 L2 范数平方
    for i in range(A.shape[0]):
        IP = np.square(np.linalg.norm(A[i]))  # 当前行向量的 L2 范数平方
        IP_0 += IP
    # 计算平均值的倒数作为带宽参数
    lambd = 1 / ((1 / A.shape[0]) * IP_0)
    return lambd


def calculate_GaussianKernel_sim(A):
    """
    基于关联谱 A 计算高斯核相似度矩阵
    
    参数:
        A: 关联矩阵，用于计算高斯核相似度
        
    返回:
        gauss_kernel_sim: 高斯核相似度矩阵，其中每个元素表示对应行向量间的高斯核相似度
    """
    # 获取高斯核带宽参数
    kernel_bandwidth = calculate_kernel_bandwidth(A)
    # 初始化高斯核相似度矩阵
    gauss_kernel_sim = np.zeros((A.shape[0], A.shape[0]))
    # 遍历所有行向量对，计算其高斯核相似度
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            # 根据高斯核函数公式计算相似度
            gaussianKernel = np.exp(-kernel_bandwidth * np.square(np.linalg.norm(A[i] - A[j])))
            gauss_kernel_sim[i][j] = gaussianKernel
    return gauss_kernel_sim


# ========= 功能相似度（PBPA） =========
def PBPA(RNA_i, RNA_j, di_sim, rna_di):
    """
    计算两个RNA分子之间的功能相似度
    
    该方法基于RNA与疾病的关联信息以及疾病间的相似度信息，通过计算两个RNA分子
    关联的疾病集合间的最大相似度来评估它们的功能相似性。
    
    参数:
        RNA_i (int): 第一个RNA分子的索引
        RNA_j (int): 第二个RNA分子的索引
        di_sim (numpy.ndarray): 疾病相似度矩阵，形状为(疾病数, 疾病数)
        rna_di (numpy.ndarray): RNA-疾病关联矩阵，形状为(RNA数, 疾病数)
        
    返回:
        float: 两个RNA分子之间的功能相似度值，范围在[0,1]之间。
               如果其中一个RNA没有关联的疾病，则返回0
    """
    diseaseSet_i = rna_di[RNA_i] > 0
    diseaseSet_j = rna_di[RNA_j] > 0
    diseaseSim_ij = di_sim[diseaseSet_i][:, diseaseSet_j]
    ijshape = diseaseSim_ij.shape
    if ijshape[0] == 0 or ijshape[1] == 0:
        return 0
    return (sum(np.max(diseaseSim_ij, axis=0)) + sum(np.max(diseaseSim_ij, axis=1))) / (ijshape[0] + ijshape[1])


def getRNA_functional_sim(RNAlen, diSiNet, rna_di):
    """
    构建 RNA 功能相似度网络（对称矩阵，对角线为 1）
    
    使用 PBPA 算法计算所有 RNA 分子两两之间的功能相似度，构建完整的相似度矩阵。
    矩阵是对称的，对角线元素为 1（表示每个RNA与自身的相似度）。
    
    参数:
        RNAlen (int): RNA分子的数量
        diSiNet (numpy.ndarray): 疾病相似度矩阵，形状为(疾病数, 疾病数)
        rna_di (numpy.ndarray): RNA-疾病关联矩阵，形状为(RNA数, 疾病数)
        
    返回:
        numpy.ndarray: RNA功能相似度网络矩阵，形状为(RNA数, RNA数)
                      矩阵是对称的，对角线元素为1
    """
    # 初始化RNA相似度矩阵
    RNASiNet = np.zeros((RNAlen, RNAlen))
    
    # 只计算上三角矩阵，避免重复计算
    for i in range(RNAlen):
        for j in range(i + 1, RNAlen):
            # 使用PBPA算法计算RNA功能相似度，并同时设置对称位置的值
            RNASiNet[i, j] = RNASiNet[j, i] = PBPA(i, j, diSiNet, rna_di)
    
    # 将对角线元素设为1（自相似度）
    RNASiNet = RNASiNet + np.eye(RNAlen)  # 自相似度设为 1
    return RNASiNet


# ========= 标签二值化与相似度融合 =========
def label_preprocess(sim_matrix):
    """
    对输入的相似度矩阵进行二值化处理。

    该函数将相似度矩阵中的每个元素与阈值 0.8 进行比较，
    大于或等于 0.8 的元素被置为 1，其余置为 0。

    Parameters
    ----------
    sim_matrix : numpy.ndarray
        输入的相似度矩阵，通常为对称矩阵，元素值范围在 [0, 1] 之间。

    Returns
    -------
    new_sim_matrix : numpy.ndarray
        二值化后的相似度矩阵，形状与输入矩阵相同，仅包含 0 和 1。
    """
    # 创建与输入矩阵形状相同的结果矩阵，初始值全为0
    new_sim_matrix = np.zeros(shape=sim_matrix.shape)
    # 遍历矩阵中的每个元素进行二值化处理
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            if sim_matrix[i][j] >= 0.8:
                new_sim_matrix[i][j] = 1
    return new_sim_matrix


def RNA_fusion_sim(G1, G2, F, threshold=0.1):
    """
    融合两种RNA高斯相似度与功能相似度并进行预处理
    
    该函数首先计算两个高斯相似度矩阵的平均值，然后根据给定阈值决定
    使用功能相似度还是平均高斯相似度作为最终结果，最后对融合后的
    相似度矩阵进行标签预处理。
    
    参数:
        G1: 第一个RNA高斯相似度矩阵
        G2: 第二个RNA高斯相似度矩阵
        F: RNA功能相似度矩阵
        threshold: 阈值，默认为0.1，用于判断使用功能相似度还是高斯相似度的平均值
        
    返回:
        fusion_sim: 融合后的RNA相似度矩阵，经过标签预处理后的结果
    """
    fusion_sim = np.zeros((len(G1), len(G2)))
    G = (G1 + G2) / 2
    for i in range(len(G1)):
        for j in range(len(G1)):
            if F[i][j] > threshold:  # 使用阈值而非简单的 >0
                fusion_sim[i][j] = F[i][j]
            else:
                fusion_sim[i][j] = G[i][j]
    fusion_sim = label_preprocess(fusion_sim)
    return fusion_sim


def dis_fusion_sim(G1, G2, SD):
    """
    融合两种疾病高斯相似度与语义相似度：先均值再二值化
    
    该函数将两种高斯相似度和一种语义相似度进行融合处理，首先计算
    两种高斯相似度的平均值，然后将其与语义相似度再次平均，最后
    对结果进行二值化处理得到最终的融合相似度矩阵。
    
    参数:
        G1: 第一种疾病高斯相似度矩阵
        G2: 第二种疾病高斯相似度矩阵
        SD: 疾病语义相似度矩阵
        
    返回:
        fusion_sim: 融合后的疾病相似度矩阵（已二值化处理）
    """
    fusion_sim = (SD + (G1 + G2) / 2) / 2
    fusion_sim = label_preprocess(fusion_sim)
    return fusion_sim


# ========= 示例入口 =========
if __name__ == '__main__':
    # ========== 数据加载部分 ==========
    # 使用 dataset1 的示例数据 - 第一次加载
    # 加载长非编码RNA-疾病关联矩阵
    lnc_dis = np.loadtxt("dataset1/lnc_dis_association.txt")
    # 加载microRNA-疾病关联矩阵
    mi_dis = np.loadtxt("dataset1/mi_dis.txt")
    # 加载lncRNA-miRNA关联矩阵
    lnc_mi = np.loadtxt("dataset1/lnc_mi.txt")
    # 加载疾病语义相似度矩阵
    dis_sem_sim = np.loadtxt("dataset1/dis_sem_sim.txt")
    # 导入日志管理器并创建日志记录器
    from log_output_manager import get_logger
    _logger = get_logger()
    # 记录加载数据的形状信息，便于调试和确认数据维度是否正确
    _logger.info(f"{lnc_dis.shape} {mi_dis.shape} {lnc_mi.shape} {dis_sem_sim.shape}")

    # 使用 dataset2 的示例数据（注意：原路径文本中使用了 dataset1，可能为笔误，保留原样）
    # 再次加载数据，覆盖之前的变量值
    lnc_dis = np.loadtxt("dataset1/lnc_dis.txt")  # 注意：这里文件名与第一次不同，缺少了"_association"
    mi_dis = np.loadtxt("dataset1/mi_dis.txt")
    lnc_mi = np.loadtxt("dataset1/lnc_mi.txt")
    dis_sem_sim = np.loadtxt("dataset1/dis_sem_sim.txt")
    # 再次记录数据形状信息
    _logger = get_logger()
    _logger.info(f"{lnc_dis.shape} {mi_dis.shape} {lnc_mi.shape} {dis_sem_sim.shape}")

    # ========== 数据预处理部分（当前未启用） ==========
    # 示例：使用全部样本进行计算（测试集置零流程保留在注释中）
    # lnc_dis_test_id = np.loadtxt("dataset1/lnc_dis_test_id1.txt")  # 加载lncRNA-疾病测试集索引
    # mi_dis_test_id = np.loadtxt("dataset1/mi_dis_test_id1.txt")    # 加载miRNA-疾病测试集索引
    # mi_lnc_test_id = np.loadtxt("dataset1/mi_lnc_test_id1.txt")    # 加载miRNA-lncRNA测试集索引
    # lnc_dis = Preproces_Data(lnc_dis, lnc_dis_test_id)  # 处理lncRNA-疾病关联矩阵，将测试集置零
    # mi_dis = Preproces_Data(mi_dis, mi_dis_test_id)      # 处理miRNA-疾病关联矩阵，将测试集置零
    # mi_lnc = Preproces_Data(lnc_mi.T, mi_lnc_test_id)    # 处理miRNA-lncRNA关联矩阵，将测试集置零

    # ========== lncRNA 相似度计算部分 ==========
    # 计算 lncRNA 相似度
    # 基于lncRNA-疾病关联计算高斯核相似度
    lnc_gau_1 = calculate_GaussianKernel_sim(lnc_dis)
    # 基于lncRNA-miRNA关联计算高斯核相似度
    lnc_gau_2 = calculate_GaussianKernel_sim(lnc_mi)
    # 计算lncRNA的功能相似度
    lnc_fun = getRNA_functional_sim(RNAlen=len(lnc_dis), diSiNet=copy.copy(dis_sem_sim), rna_di=copy.copy(lnc_dis))
    # 融合多种相似度，得到最终的lncRNA相似度矩阵
    lnc_sim = RNA_fusion_sim(lnc_gau_1, lnc_gau_2, lnc_fun)

    # ========== miRNA 相似度计算部分 ==========
    # 计算 miRNA 相似度
    # 基于miRNA-疾病关联计算高斯核相似度
    mi_gau_1 = calculate_GaussianKernel_sim(mi_dis)
    # 基于miRNA-lncRNA关联计算高斯核相似度（注意转置lnc_mi矩阵）
    mi_gau_2 = calculate_GaussianKernel_sim(lnc_mi.T)
    # 计算miRNA的功能相似度
    mi_fun = getRNA_functional_sim(RNAlen=len(mi_dis), diSiNet=copy.copy(dis_sem_sim), rna_di=copy.copy(mi_dis))
    # 融合多种相似度，得到最终的miRNA相似度矩阵
    mi_sim = RNA_fusion_sim(mi_gau_1, mi_gau_2, mi_fun)

    # ========== 疾病相似度计算部分 ==========
    # 计算疾病相似度
    # 基于lncRNA-疾病关联计算高斯核相似度（注意转置矩阵）
    dis_gau_1 = calculate_GaussianKernel_sim(lnc_dis.T)
    # 基于miRNA-疾病关联计算高斯核相似度（注意转置矩阵）
    dis_gau_2 = calculate_GaussianKernel_sim(mi_dis.T)
    # 融合高斯核相似度和语义相似度，得到最终的疾病相似度矩阵
    dis_sim = dis_fusion_sim(dis_gau_1, dis_gau_2, dis_sem_sim)