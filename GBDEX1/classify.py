import numpy as np  # 导入 numpy 模块

def kmeans_clustering(elements, K=2, LoopN=100, Tolerance=1e-4, width=0, height=0):
    """
    多波段遥感图像K-means分类核心算法
    :param height:
    :param width:
    :param elements: N行M列的numpy数组，每个列代表一个像素的M个波段值
    :param K: 分类个数
    :param LoopN: 最大迭代次数
    :param Tolerance: 中心点变化容差
    """

    assert elements.ndim == 2, "输入的elements必须是二维数组"
    assert width * height == 0 or elements.shape[1] == width * height, "输入的elements与输入的高度宽度不匹配"

    # 创建结果数组（10行 x (height*width)列）
    XYR_table = []
    if width * height != 0:
        XYR_table = np.zeros((3, height * width))
        XYR_table[0] = np.tile(np.arange(width), height)  # 列号按行重复
        XYR_table[1] = np.repeat(np.arange(height), width)  # 行号逐行展开

    # 初始化中心点（K-means++优化版）
    centers = initialize_centers(elements, K)
    labels = np.zeros(height * width)
    # 主循环
    for i in range(LoopN):
        print("第" + str(i) + "次K-means循环")
        # 步骤1：分配标签
        labels = assign_labels(elements, centers)
        if width * height != 0:
            XYR_table[2] = labels  # 更新结果表的第三行

        # 步骤2：计算新质心
        new_centers = calculate_centroids(elements, labels, K)

        # 步骤3：检查终止条件
        if np.max(np.linalg.norm(new_centers - centers, axis=1)) < Tolerance:
            break
        centers = new_centers

    print("分类完成")
    # 将结果转换为二维数组
    resultImg = []
    if width * height != 0:
        resultImg = convert_to_2d(XYR_table)
        resultImg = np.flipud(resultImg)
        resultImg = [list(row) for row in zip(*resultImg[::-1])]
    return resultImg, centers, labels


def initialize_centers(elements, K):
    """K-means++初始化中心点"""
    N = elements.shape[1]
    centers = np.zeros((K, elements.shape[0]))

    # 随机选择第一个中心点
    first_idx = np.random.randint(N)
    centers[0] = elements[:, first_idx]

    # 依次选择后续中心点
    for i in range(1, K):
        # 计算每个点到最近现有中心的距离
        distances = np.min(np.linalg.norm(
            elements.T[:, np.newaxis, :] - centers[:i],
            axis=2
        ), axis=1)
        # 选择距离最远的点作为新中心
        new_center_idx = np.argmax(distances)
        centers[i] = elements[:, new_center_idx]
    print("中心点初始化完成")
    return centers


def assign_labels(elements, centers):
    """分配样本到最近的中心点"""
    # 向量化计算所有距离
    distances = np.linalg.norm(
        elements.T[:, np.newaxis, :] - centers,
        axis=2
    )
    return np.argmin(distances, axis=1)


def calculate_centroids(elements, labels, K):
    """计算各簇的质心"""
    new_centers = np.zeros((K, elements.shape[0]))
    for k in range(K):
        # 获取当前簇的所有样本
        cluster_samples = elements[:, labels == k]
        if cluster_samples.size == 0:  # 处理空簇
            new_centers[k] = elements[:, np.random.choice(elements.shape[1])]
        else:
            new_centers[k] = np.mean(cluster_samples, axis=1)
    return new_centers


def convert_to_2d(arr_3d):
    """
    将三行数组转换为二维数组
    :param arr_3d: 输入的三行numpy数组，前两行为坐标，第三行为数据
    :return: 二维数组矩阵
    """
    # 提取坐标和数据
    x_coords = arr_3d[0].astype(int)
    y_coords = arr_3d[1].astype(int)
    data = arr_3d[2]

    # 计算矩阵尺寸
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # 创建空矩阵（默认填充0）
    matrix = np.zeros((x_max - x_min + 1, y_max - y_min + 1), dtype=data.dtype)

    # 填充数据（处理坐标重复情况）
    for x, y, val in zip(x_coords, y_coords, data):
        matrix[x - x_min, y - y_min] = val
    return matrix


def getDistortions(data, krange):
    """计算不同k值下k-means聚类结果的distortion指标

    参数：
    data : numpy.ndarray
        输入数据矩阵，形状为(dimensions, samples)，每列代表一个样本数据点
    krange : iterable
        需要测试的k值组成的可迭代对象(如列表/范围)

    返回值：
    list[float]
        每个k值对应的平均distortion值，与krange顺序一致
    """
    dists = []
    for k in krange:
        print("正在计算k=" + str(k) + "的 distortion")

        # 执行k-means聚类算法，获取聚类中心和标签
        result_img, centers, labels = kmeans_clustering(
            elements=data,
            K=k,
            LoopN=100,  # 适当增大迭代次数
            Tolerance=1e-4,  # 容差阈值
        )

        # 手动计算所有数据点到对应聚类中心的平均距离
        total_dist = 0.0
        for i in range(data.shape[1]):
            cluster_center = centers[int(labels[i])]
            dist = np.linalg.norm(data[:,i] - cluster_center)
            total_dist += dist
        print("k=" + str(k) + " distortion=" + str(total_dist / data.shape[1]))
        dists.append(total_dist / data.shape[1])

    return dists
