import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import jenkspy

def exponential_compression(data, rootV=1):
    """
    对数据进行指数压缩处理

    Args:
        data (iterable): 输入数据序列
        rootV (int, optional): 指数压缩的根值参数，默认为1（不压缩）

    Returns:
        list: 压缩后的数据列表

    Raises:
        ValueError: 当rootV <= 0时抛出异常
    """
    if rootV <= 0:
        raise ValueError("rootV必须为大于0的整数")

    # 指数压缩计算公式示例
    return [x ** (1/rootV) if x >= 0 else -abs(x) ** (1/rootV) for x in data]

def KMCluster(data, StrID, StrCluster, Knum, label_str=None, ignoreV=None, rootV=1):
    """
    执行带数据预处理和标签映射的K-means聚类分析

    参数:
    data: DataFrame
        包含待聚类数据的源数据集
    StrID: str
        唯一标识列的列名，用于校验数据完整性
    StrCluster: str
        需要进行聚类分析的数值列名
    Knum: int
        期望的总聚类数（包含忽略值类别）
    label_str: list[str], optional
        自定义聚类标签列表，长度需与有效聚类数匹配
    ignoreV: float, optional
        需要单独过滤的特定值（将被归为单独类别）
    rootV: int, optional
        指数压缩系数，默认为1表示不压缩

    返回值:
    DataFrame
        包含原始ID和聚类结果的两列数据集

    异常:
    ValueError
        当存在重复ID或标签数量不匹配时抛出
    """
    # 参数校验
    # 检查ID列唯一性约束
    if data[StrID].duplicated().any():
        raise ValueError("存在重复的ID值")
    # 验证标签数量与聚类数的逻辑关系
    if label_str and (len(label_str) != Knum - (1 if ignoreV else 0)):
        raise ValueError("label_mapping长度需与有效聚类数一致")

    # 数据预处理
    # 创建数据副本并进行指数变换
    df = data[[StrID, StrCluster]].copy()
    if rootV != 1:
        df[StrCluster] = exponential_compression(df[StrCluster], rootV)

    # 忽略值处理
    # 构建有效数据掩码和调整后的聚类数
    mask = np.ones(len(df), dtype=bool)
    if ignoreV is not None:
        mask = (df[StrCluster] != ignoreV)
        valid_data = df.loc[mask, StrCluster].values.reshape(-1, 1)
        n_clusters = Knum - 1
    else:
        valid_data = df[StrCluster].values.reshape(-1, 1)
        n_clusters = Knum

    # 核心聚类执行
    # 使用自动初始化配置的K-means算法
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(valid_data)

    # 结果列初始化
    # 创建带掩码的聚类结果列
    df[StrCluster+'_judge'] = None
    df.loc[mask, StrCluster+'_judge'] = kmeans.labels_

    # 标签映射处理
    # 根据聚类中心排序建立标签映射关系
    cluster_order = kmeans.cluster_centers_.argsort(axis=0).ravel()
    label_mapping = {}
    # 含忽略值的标签分配逻辑
    if label_str:
        if ignoreV is not None:
            for i in range(len(cluster_order)):
                label_mapping[cluster_order[i]] = label_str[i+1]
            label_mapping[None] = label_str[0]  # 处理忽略值标签
            df[StrCluster+'_judge'] = df[StrCluster+'_judge'].map(label_mapping)
            df.loc[~mask, StrCluster+'_judge'] = label_str[0]
        else:
            for i in range(len(cluster_order)):
                label_mapping[cluster_order[i]] = label_str[i]
            df[StrCluster + '_judge'] = df[StrCluster + '_judge'].map(label_mapping)

    # 结果分析输出
    # 准备聚类结果统计信息
    clustered_data = df[mask].copy()
    clustered_data['cluster'] = kmeans.labels_

    print("\n\n聚类结果（K-means）：")
    print("指数压缩值：" + str(rootV))

    # 忽略值信息输出
    if ignoreV is not None:
        ignore_count = (~mask).sum()
        print(str(label_str[0]) + f" （{ignore_count}条）： {ignoreV}")
    else:
        print("无忽略值")

    # 区间统计输出
    # 按中心点升序输出各簇统计信息
    sorted_clusters = np.argsort(kmeans.cluster_centers_.flatten())
    print("区间（升序排列）：")
    for order, cluster_idx in enumerate(sorted_clusters, start=1):
        cluster_data = clustered_data[clustered_data['cluster'] == cluster_idx][StrCluster]
        min_val = cluster_data.min()
        max_val = cluster_data.max()
        count = cluster_data.shape[0]

        # 显示序号调整逻辑
        display_order = order if not ignoreV else order-1
        if ignoreV is not None:
            display_order = max(0, display_order)

        # 标签名称生成逻辑
        if label_str:
            label_name = label_str[order] if ignoreV else label_str[order-1]
        else:
            label_name = f"类别 {display_order}"

        # 显示反压缩后的实际数值区间
        print(f"{label_name}（{count}条）区间: [{min_val**rootV:.2f}, {max_val**rootV:.2f}]")

    return df[[StrID, StrCluster+'_judge']]


def JenkCluster(data, StrID, StrCluster, Knum, label_str=None, ignoreV=None, rootV=1):
    """
    使用Jenks自然断裂法对数据进行聚类分析

    参数:
        data (DataFrame): 原始数据集
        StrID (str): 标识列的名称（需唯一）
        StrCluster (str): 需要聚类的数值列名称
        Knum (int): 期望的聚类总数（包含忽略值类别）
        label_str (List[str], optional): 聚类标签列表。长度需等于Knum（有忽略值时等于Knum-1）
        ignoreV (optional): 需要忽略的特殊值，被忽略值将单独归类
        rootV (float): 指数压缩系数，用于数据预处理（默认1表示不压缩）

    返回:
        DataFrame: 包含ID列和聚类结果列的二维数据表

    异常:
        ValueError: 参数校验失败或算法执行错误时抛出
    """
    # 参数校验
    # 检查ID列是否唯一
    if data[StrID].duplicated().any():
        raise ValueError("存在重复的ID值")

    # 数据预处理
    # 创建数据副本并进行指数压缩
    df = data[[StrID, StrCluster]].copy()
    if rootV != 1:
        df[StrCluster] = exponential_compression(df[StrCluster], rootV)

    # 忽略值处理
    # 构建有效数据掩码并调整实际聚类数
    mask = np.ones(len(df), dtype=bool)
    if ignoreV is not None:
        mask = (df[StrCluster] != ignoreV)
        n_clusters = Knum - 1
    else:
        n_clusters = Knum

    # 数据有效性验证
    # 检查有效数据的多样性
    valid_data = df.loc[mask, StrCluster].values.astype(float)
    if len(np.unique(valid_data)) < 2:
        raise ValueError("Jenks算法需要至少2个不同数值")
    if label_str and (len(label_str) != Knum - (1 if ignoreV else 0)):
        raise ValueError("标签数量与聚类数不匹配")

    # Jenks算法核心执行
    # 获取自然断裂点
    try:
        breaks = jenkspy.jenks_breaks(valid_data, n_classes=n_clusters)
    except Exception as e:
        raise ValueError(f"Jenks算法执行失败: {str(e)}")

    # 标签分配逻辑
    # 使用断裂点进行离散化分箱
    labels = np.searchsorted(breaks, valid_data, side='right') - 1
    labels = np.clip(labels, 0, n_clusters - 1)  # 确保极端值落在有效区间内

    # 结果整合
    # 将聚类结果合并到原始数据
    df[StrCluster + '_judge'] = None
    df.loc[mask, StrCluster + '_judge'] = labels

    # 标签映射处理
    # 将数字标签转换为语义化标签
    if label_str:
        if ignoreV is not None:
            label_mapping = {i: label_str[i + 1] for i in range(n_clusters)}
            label_mapping[None] = label_str[0]  # 特殊处理忽略值标签
        else:
            label_mapping = {i: label_str[i] for i in range(n_clusters)}

        df[StrCluster + '_judge'] = df[StrCluster + '_judge'].map(label_mapping)
        if ignoreV is not None:
            df.loc[~mask, StrCluster + '_judge'] = label_str[0]

    # 结果展示
    # 输出格式化统计信息
    print("\n\n聚类结果（Jenks自然断裂法）：")
    print("指数压缩值："+str(rootV))

    if ignoreV is not None:
        ignore_count = (~mask).sum()
        print(str(label_str[0])+f" （{ignore_count}条）： {ignoreV}")
    else:
        print("无忽略值")

    # 区间统计输出
    # 格式化显示各区间范围和样本数量
    print("区间（升序排列）：")
    sorted_breaks = sorted(breaks)
    for i in range(len(sorted_breaks) - 1):
        lower = sorted_breaks[i]
        upper = sorted_breaks[i + 1]

        # 区间计数逻辑
        if i == 0:
            count = ((valid_data >= lower) & (valid_data <= upper)).sum()
        else:
            count = ((valid_data > lower) & (valid_data <= upper)).sum()

        # 标签处理逻辑
        if label_str:
            if ignoreV is not None:
                label = label_str[i + 1] if (i + 1) < len(label_str) else f"区间{i + 1}"
            else:
                label = label_str[i] if i < len(label_str) else f"区间{i + 1}"
        else:
            label = f"区间{i + 1}"

        # 区间显示格式处理
        if i == len(sorted_breaks) - 2:
            interval_str = f"[{lower**rootV:.2f}, {upper**rootV:.2f}]"  # 闭合区间显示
        else:
            interval_str = f"({lower**rootV:.2f}, {upper**rootV:.2f}]"

        print(f"{label}（{count}条）: {interval_str}")

    return df[[StrID, StrCluster + '_judge']]


def SliptClusters(data, StrID, StrCluster, SplitValues, label_str):
    """
    根据分割值分类目标字段并打标签

    参数：
    data - 原始数据集（DataFrame）
    StrID - 唯一标识字段名（字符串）
    StrCluster - 需要分类的目标字段名（字符串）
    SplitValues - 分割值数组（排序后的数值列表）
    label_str - 分类标签数组（长度需为len(SplitValues)+1）

    返回：
    包含StrID和分类结果的DataFrame
    """
    # 生成区间边界
    bins = [-np.inf] + sorted(SplitValues) + [np.inf]

    # 生成区间描述字符串
    interval_strings = []
    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]
        # 处理无穷大显示
        left_str = '-∞' if left == -np.inf else f'{left:.2f}'
        right_str = '+∞' if right == np.inf else f'{right:.2f}'
        interval_str = f'[{left_str}, {right_str})'  # 左闭右开
        interval_strings.append(interval_str)

    # 分箱操作
    data['category'] = pd.cut(data[StrCluster], bins=bins, labels=label_str, right=False)

    # 按标签顺序统计（确保顺序与label_str一致）
    counts = data['category'].value_counts().reindex(label_str, fill_value=0)

    # 增强版统计输出
    print("\n\n区间分布统计：")
    for label, interval, count in zip(label_str, interval_strings, counts):
        print(f"{label}（{interval}）: {count}个")

    # 返回指定字段
    return data[[StrID, 'category']].rename(columns={'category': StrCluster+'_judge'})
