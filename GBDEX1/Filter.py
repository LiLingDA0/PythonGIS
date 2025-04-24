import numpy as np  # 导入 numpy 模块

def normalize(image, new_min=0, new_max=1, ifInt=False):
    """
    对输入图像进行线性归一化处理

    参数:
        image (ndarray): 输入的图像数组
        new_min (float): 归一化后的最小值，默认0
        new_max (float): 归一化后的最大值，默认1
        ifInt (bool): 是否返回整数类型，若为True则结果会四舍五入并调整最大值

    返回:
        ndarray: 归一化后的图像数组。当ifInt=True时返回整数数组，否则返回浮点数组
    """
    old_min = image.min()
    old_max = image.max()
    if ifInt:
        new_max -=1  # 调整最大值范围用于整数转换
        normalized_image = (image - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        return np.round(normalized_image)
    else:
        normalized_image = (image - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        return normalized_image

# 一级滤波模式字典，包含基本统计量计算方式
FilterMode1={
    'mean': lambda image: np.mean(image),    # 计算局部均值
    'median': lambda image: np.median(image) # 计算局部中值
}

# 二级滤波模式字典，包含需要特殊处理的边缘检测算子
FilterMode2={
    "sobel": lambda image,stretch: sobel_filter(image,stretch),  # Sobel算子边缘检测
    "roberts": lambda image,stretch: roberts_filter(image,stretch)  # Roberts算子边缘检测
}

def f_mode(image, mode='mean', size=3, stretch=1):
    """
    通用滤波处理函数，支持多种滤波模式

    参数:
        image (ndarray): 输入的单通道灰度图像
        mode (str): 滤波模式，可选 'mean','median','sobel','roberts'
        size (int): 滤波窗口尺寸（仅一级滤波模式有效）
        stretch (float): 结果拉伸系数（仅二级滤波模式有效）

    返回:
        ndarray/str: 滤波后的图像数组，或错误提示字符串

    实现说明:
        - 一级滤波使用滑动窗口进行局部统计计算
        - 二级滤波调用专用边缘检测算子处理
    """
    if mode in FilterMode1:
        # 一级滤波处理流程
        Func = FilterMode1[mode]
        pad_width = size // 2
        padded_image = np.pad(image, pad_width, mode='symmetric')  # 对称填充边界
        filtered_image = np.zeros_like(image)

        # 滑动窗口遍历每个像素
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                localImg = padded_image[i:i + size, j:j + size]
                filtered_image[i, j] = Func(localImg)
        return filtered_image
    elif mode in FilterMode2:
        # 二级滤波直接调用对应算子
        Func = FilterMode2[mode]
        return Func(image,stretch)
    else:
        return "不支持的计算模式"


def roberts_filter(image, stretch=1):
    """
    Roberts交叉算子边缘检测

    参数:
        image (ndarray): 输入的单通道灰度图像
        stretch (float): 结果增强系数

    返回:
        ndarray: 边缘检测结果图像
    """
    filterX = np.array([[-1, 0], [0, 1]])  # Roberts水平模板
    filterY = np.array([[0, -1], [1, 0]])  # Roberts垂直模板
    return custom_filter2(image, filterX, filterY,mode='absSum', stretch=stretch)


def sobel_filter(image, stretch=1):
    """
    Sobel算子边缘检测

    参数:
        image (ndarray): 输入的单通道灰度图像
        stretch (float): 结果增强系数

    返回:
        ndarray: 边缘检测结果图像
    """
    filterX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel水平模板
    filterY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Sobel垂直模板
    return custom_filter2(image, filterX, filterY,mode='absSum', stretch=stretch)


def custom_filter(image, matrix, stretch=1):
    """
    单矩阵自定义卷积滤波器

    参数:
        image (ndarray): 输入的单通道灰度图像
        matrix (ndarray): 二维卷积核矩阵
        stretch (float): 结果增强系数

    返回:
        ndarray/str: 滤波结果数组或错误提示

    实现说明:
        - 自动计算所需填充宽度
        - 使用对称填充处理边界
    """
    if matrix.shape[0]!=matrix.shape[1]:  # 检查矩阵是否为正方形
        return "滤波器矩阵必须为正方形"
    if matrix.shape[0]!=2:
        pad_width = matrix.shape[0] // 2
    else:
        pad_width = ((0, 1), (0, 1))  # 2x2矩阵特殊填充处理
    padded_image = np.pad(image, pad_width, mode='symmetric')
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            localImg = padded_image[i:i+matrix.shape[0], j:j+matrix.shape[1]]
            filtered_image[i, j] = np.sum(localImg * matrix)* stretch  # 卷积运算
    return filtered_image


# 双矩阵滤波结果合并方式字典
CustomF2Mode={
    'mean': lambda MA , MB: (MA+MB)/2,  # 均值合并
    'max': lambda MA , MB: np.max([MA,MB]),  # 取最大值
    'min': lambda MA , MB: np.min([MA,MB]),  # 取最小值
    'sum': lambda MA , MB: MA+MB,  # 直接求和
    'avg': lambda MA , MB: (MA+MB)/2,  # 平均值（同mean）
    'absSum': lambda MA , MB: np.sqrt(MA**2+MB**2)  # 平方和开根号
}

def custom_filter2(image, matrixA,matrixB,mode = 'mean', stretch=1):
    """
    双矩阵自定义卷积滤波器

    参数:
        image (ndarray): 输入的单通道灰度图像
        matrixA (ndarray): 第一个二维卷积核
        matrixB (ndarray): 第二个二维卷积核
        mode (str): 双通道结果合并方式
        stretch (float): 结果增强系数

    返回:
        ndarray/str: 滤波结果数组或错误提示

    实现说明:
        - 要求两个矩阵为相同尺寸的正方形
        - 支持多种双通道计算结果合并方式
    """
    if matrixA.shape[0]!=matrixA.shape[1] or matrixB.shape[0]!=matrixB.shape[1]:
        return "滤波器矩阵必须为正方形"
    if matrixA.shape[0]!=matrixB.shape[0]:
        return "两个矩阵大小要相等"
    if matrixA.shape[0]!=2:
        pad_width = matrixA.shape[0] // 2
    else:
        pad_width = ((0, 1), (0, 1))  # 2x2矩阵特殊填充处理
    padded_image = np.pad(image, pad_width, mode='symmetric')
    filtered_image = np.zeros_like(image)
    step = matrixA.shape[0]
    if mode in CustomF2Mode:
        Func = CustomF2Mode[mode]
    else:
        return "不支持的计算模式"

    # 双矩阵卷积计算循环
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            localImg = padded_image[i:i+step, j:j+step]
            MA = np.sum(localImg * matrixA)  # 矩阵A的卷积结果
            MB = np.sum(localImg * matrixB)  # 矩阵B的卷积结果
            filtered_image[i, j] = Func(MA,MB) * stretch  # 合并计算结果并增强

    return filtered_image
