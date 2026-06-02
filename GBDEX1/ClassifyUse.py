import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os  # 导入 os 模块
import numpy as np  # 导入 numpy 模块
import Filter
from classify import kmeans_clustering


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

    resultImg=kmeans_clustering(Elements, 5, 100, 0.001, width, height)[0]

            # 保存目录
    output_dir = os.path.join(project_path, 'RESULT')  # 创建结果目录
    os.makedirs(output_dir, exist_ok=True)  # 自动创建目录

    # 修改保存部分的代码为：
    if isinstance(resultImg, list):  # 添加类型检查
        resultImg = np.array(resultImg)  # 转换为NumPy数组

    # 确保数组维度与原始影像一致
    resultImg = resultImg.reshape((height, width))  # 调整为(height, width)形状
    meta.update({
        'count': 1,  # 单波段
        'dtype': 'uint8',  # 根据实际数据类型调整
        'nodata': None  # 如果没有无效值
    })

    # 添加数据类型验证
    print("结果数组类型:", resultImg.dtype)  # 调试输出
    print("元数据类型要求:", meta['dtype'])  # 调试输出

    # 保存时显式指定数据类型
    with rasterio.open(os.path.join(output_dir, 'classification.tif'), 'w', **meta) as dst:
        dst.write(resultImg.astype(meta['dtype']), 1)

    print("开始加载图像")
    # 配置 matplotlib 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体或其他支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 解负号显示问题
    unique_values = np.unique(resultImg)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_values)))

    # 创建离散色彩映射
    cmap = ListedColormap(colors)
    norm = plt.Normalize(vmin=unique_values.min() - 0.5,
                         vmax=unique_values.max() + 0.5)

    rgbImg = np.dstack((
       Filter.normalize( bands_data[3]),
        Filter.normalize(bands_data[2]),
        Filter.normalize(bands_data[1]),
    ))

    fig, axes = plt.subplots(1, 2)  # 创建1行2列布局
    plt.tight_layout()
    axes[0].imshow(normalize_image(rgbImg))
    axes[0].set_title('原始图像')
    axes[1].imshow(resultImg, cmap=cmap, norm=norm)
    axes[1].set_title('KMeans分类结果')

    # 保存整张对比图
    fig.savefig(os.path.join(output_dir, 'classification_comparison.png'),
                dpi=300, bbox_inches='tight')

    # 保存分类结果图（修正后的参数）
    plt.imsave(
        os.path.join(output_dir, 'classification.png'),
        resultImg,
        cmap=cmap,
        vmin=unique_values.min() - 0.5,
        vmax=unique_values.max() + 0.5,
        origin='upper'
    )

    plt.show()