import numpy as np  # 数值计算与数组/矩阵操作
import copy  # 用于创建对象副本（浅拷贝/深拷贝均可）


# ========= 预处理与核带宽 =========
def Preproces_Data(A, test_id):
    """将测试集中的阳性样本在关联矩阵 A 中置 0（不改动原始矩阵）"""
    copy_A = A / 1  # 创建 A 的副本，避免修改原数据
    for i in range(test_id.shape[0]):
        copy_A[int(test_id[i][0])][int(test_id[i][1])] = 0
    return copy_A


def calculate_kernel_bandwidth(A):
    """计算高斯核带宽参数（基于每行谱的 L2 范数平方的平均值的倒数）"""
    IP_0 = 0
    for i in range(A.shape[0]):
        IP = np.square(np.linalg.norm(A[i]))  # 当前行向量的 L2 范数平方
        IP_0 += IP
    lambd = 1 / ((1 / A.shape[0]) * IP_0)
    return lambd


def calculate_GaussianKernel_sim(A):
    """基于关联谱 A 计算高斯核相似度矩阵"""
    kernel_bandwidth = calculate_kernel_bandwidth(A)
    gauss_kernel_sim = np.zeros((A.shape[0], A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            gaussianKernel = np.exp(-kernel_bandwidth * np.square(np.linalg.norm(A[i] - A[j])))
            gauss_kernel_sim[i][j] = gaussianKernel
    return gauss_kernel_sim


# ========= 功能相似度（PBPA） =========
def PBPA(RNA_i, RNA_j, di_sim, rna_di):
    """
    计算两个 RNA（i, j）的功能相似度：
    - 取各自关联疾病集合的子矩阵
    - 分别对两方向求最大相似度再求和，并按集合大小归一化
    """
    diseaseSet_i = rna_di[RNA_i] > 0
    diseaseSet_j = rna_di[RNA_j] > 0
    diseaseSim_ij = di_sim[diseaseSet_i][:, diseaseSet_j]
    ijshape = diseaseSim_ij.shape
    if ijshape[0] == 0 or ijshape[1] == 0:
        return 0
    return (sum(np.max(diseaseSim_ij, axis=0)) + sum(np.max(diseaseSim_ij, axis=1))) / (ijshape[0] + ijshape[1])


def getRNA_functional_sim(RNAlen, diSiNet, rna_di):
    """构建 RNA 功能相似度网络（对称矩阵，对角线为 1）"""
    RNASiNet = np.zeros((RNAlen, RNAlen))
    for i in range(RNAlen):
        for j in range(i + 1, RNAlen):
            RNASiNet[i, j] = RNASiNet[j, i] = PBPA(i, j, diSiNet, rna_di)
    RNASiNet = RNASiNet + np.eye(RNAlen)  # 自相似度设为 1
    return RNASiNet


# ========= 标签二值化与相似度融合 =========
def label_preprocess(sim_matrix):
    """将相似度矩阵按阈值二值化：>=0.8 置为 1，否则为 0"""
    new_sim_matrix = np.zeros(shape=sim_matrix.shape)
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            if sim_matrix[i][j] >= 0.8:
                new_sim_matrix[i][j] = 1
    return new_sim_matrix


def RNA_fusion_sim(G1, G2, F, threshold=0.1):
    """
    ✅ 修复6：优化融合逻辑，避免稀疏 F 导致全 0 偏置
    融合两种高斯相似度与功能相似度：
    - 当 F[i][j] > threshold 时，优先采用 F[i][j]
    - 否则取 (G1+G2)/2
    - 最后二值化处理
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
    """融合两种疾病高斯相似度与语义相似度：先均值再二值化"""
    fusion_sim = (SD + (G1 + G2) / 2) / 2
    fusion_sim = label_preprocess(fusion_sim)
    return fusion_sim


# ========= 示例入口 =========
if __name__ == '__main__':
    # 使用 dataset1 的示例数据
    lnc_dis = np.loadtxt("dataset1/lnc_dis_association.txt")
    mi_dis = np.loadtxt("dataset1/mi_dis.txt")
    lnc_mi = np.loadtxt("dataset1/lnc_mi.txt")
    dis_sem_sim = np.loadtxt("dataset1/dis_sem_sim.txt")
    from log_output_manager import get_logger
    _logger = get_logger()
    _logger.info(f"{lnc_dis.shape} {mi_dis.shape} {lnc_mi.shape} {dis_sem_sim.shape}")

    # 使用 dataset2 的示例数据（注意：原路径文本中使用了 dataset1，可能为笔误，保留原样）
    lnc_dis = np.loadtxt("dataset1/lnc_dis.txt")
    mi_dis = np.loadtxt("dataset1/mi_dis.txt")
    lnc_mi = np.loadtxt("dataset1/lnc_mi.txt")
    dis_sem_sim = np.loadtxt("dataset1/dis_sem_sim.txt")
    _logger = get_logger()
    _logger.info(f"{lnc_dis.shape} {mi_dis.shape} {lnc_mi.shape} {dis_sem_sim.shape}")

    # 示例：使用全部样本进行计算（测试集置零流程保留在注释中）
    # lnc_dis_test_id = np.loadtxt("dataset1/lnc_dis_test_id1.txt")
    # mi_dis_test_id = np.loadtxt("dataset1/mi_dis_test_id1.txt")
    # mi_lnc_test_id = np.loadtxt("dataset1/mi_lnc_test_id1.txt")
    # lnc_dis = Preproces_Data(lnc_dis, lnc_dis_test_id)
    # mi_dis = Preproces_Data(mi_dis, mi_dis_test_id)
    # mi_lnc = Preproces_Data(lnc_mi.T, mi_lnc_test_id)

    # 计算 lncRNA 相似度
    lnc_gau_1 = calculate_GaussianKernel_sim(lnc_dis)
    lnc_gau_2 = calculate_GaussianKernel_sim(lnc_mi)
    lnc_fun = getRNA_functional_sim(RNAlen=len(lnc_dis), diSiNet=copy.copy(dis_sem_sim), rna_di=copy.copy(lnc_dis))
    lnc_sim = RNA_fusion_sim(lnc_gau_1, lnc_gau_2, lnc_fun)

    # 计算 miRNA 相似度
    mi_gau_1 = calculate_GaussianKernel_sim(mi_dis)
    mi_gau_2 = calculate_GaussianKernel_sim(lnc_mi.T)
    mi_fun = getRNA_functional_sim(RNAlen=len(mi_dis), diSiNet=copy.copy(dis_sem_sim), rna_di=copy.copy(mi_dis))
    mi_sim = RNA_fusion_sim(mi_gau_1, mi_gau_2, mi_fun)

    # 计算疾病相似度
    dis_gau_1 = calculate_GaussianKernel_sim(lnc_dis.T)
    dis_gau_2 = calculate_GaussianKernel_sim(mi_dis.T)
    dis_sim = dis_fusion_sim(dis_gau_1, dis_gau_2, dis_sem_sim)