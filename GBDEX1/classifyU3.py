from classify import kmeans_clustering
import numpy as np
import rasterio
import os  # 导入 os 模块
import matplotlib.pyplot as plt

def gap(data, refs=None, nrefs=20, ks=range(1, 11)):
    """
    计算Gap统计量用于确定最佳聚类数量（基于Tibshirani, Walther的Gap Statistic方法）

    参数:
        data (numpy.ndarray): 原始数据集矩阵，形状为(n_samples, n_features)
        refs (numpy.ndarray, optional): 参考数据集立方体，形状为(n_samples, n_features, n_refs)
        nrefs (int): 当refs为None时生成的参考数据集数量，默认为20
        ks (iterable): 要测试的聚类数量范围，默认为1到10

    返回值:
        tuple: 包含两个元素的元组
            - ks (list): 输入的聚类数量列表
            - gaps (numpy.ndarray): 每个k值对应的Gap统计量

    实现说明:
        1. 使用自实现的K-means算法（来自classify.py）
        2. 完全基于numpy实现参考数据集生成
        3. 保持与原始数学公式的一致性
    """
    # 参数预处理
    n_samples, n_features = data.shape
    k_means_args = {
        'LoopN': 100,
        'Tolerance': 1e-4,
        'width': 1,  # 单像素模式
        'height': n_samples
    }

    # 生成参考数据集（均匀分布）
    if refs is None:
        # 调整维度对齐：添加两个新轴使其形状为 (1, n_features, 1)
        mins = data.min(axis=0).reshape(1, -1, 1)
        maxs = data.max(axis=0).reshape(1, -1, 1)
        rands = np.random.random((data.shape[0], data.shape[1], nrefs))
        rands = rands * (maxs - mins) + mins  # 现在可以正确广播
    else:
        rands = refs

    gaps = np.zeros(len(ks))

    # 主计算逻辑
    for i, k in enumerate(ks):
        print(f'计算Gap值，当前聚类数量：{k}')
        # 原始数据聚类
        labels = kmeans_clustering(data.T, K=k, **k_means_args)[2]

        # 计算原始离散度（使用numpy优化）
        disp = np.sum([np.linalg.norm(data - data[labels == li].mean(axis=0), axis=1).sum()
                      for li in range(k)])

        # 参考数据计算
        ref_disps = np.zeros(nrefs)
        for j in range(nrefs):
            print(f'计算参考Gap值，当前聚类数量：{k}, 当前参考数据集：{j}')
            ref_data = rands[:, :, j]
            ref_labels = kmeans_clustering(ref_data.T, K=k, **k_means_args)
            ref_disps[j] = np.sum([np.linalg.norm(ref_data - ref_data[ref_labels == li].mean(axis=0), axis=1).sum()
                                  for li in range(k)])

        # Gap值计算
        gaps[i] = np.log(ref_disps).mean() - np.log(disp)

    return ks, gaps

def normalize_image(img, percentile=98):
    """自适应对比度拉伸"""
    vmin = np.percentile(img, (100 - percentile) / 2)
    vmax = np.percentile(img, 100 - (100 - percentile) / 2)
    return np.clip((img - vmin) / (vmax - vmin), 0, 1)

# Example usage
if __name__ == "__main__":
    # 获取当前项目路径
    project_path = os.path.dirname(os.path.abspath(__file__))

    # 定义文件路径变量
    file_path = os.path.join(project_path, 'RES', 'KMData','b')  # 使用 os.path.join 构建路径

    # 读取所有波段数据（假设处理7个波段）
    bands_data = []
    for i in range(1, 8):  # 调整为实际波段数量
        with rasterio.open(file_path + str(i) + '.tif') as src:
            bands_data.append(src.read(1))
            meta = src.meta
    bands_data = np.array(bands_data)

    width = bands_data.shape[2]
    height = bands_data.shape[1]

    Elements = np.zeros((7, height * width))
    for i in range(0, bands_data.shape[0]):
        Elements[i] = bands_data[i].ravel()

    print("数据构建完毕")

    ks,gaps=gap(Elements.T)
    print(ks)
    print(gaps)

    # 绘制肘部图
    plt.plot(ks, gaps, 'bx-')
    plt.xlabel('k')
    plt.ylabel('gap')
    plt.title('The Gap Method showing the optimal k')
    output_dir = os.path.join(project_path, 'RESULT')  # 创建结果目录
    plt.savefig(os.path.join(output_dir, 'gap_chart.png'), bbox_inches='tight', dpi=300, format='png')
    plt.show()



