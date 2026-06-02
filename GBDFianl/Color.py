import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def generate_color_spectrum(color_mapping):
    """
    生成混合色谱图，展示所有可能的颜色组合及其类型比例

    参数:
        color_mapping (dict): 颜色映射字典，格式为 {'类型名': (R, G, B)}
    """
    # 1. 创建基础图例
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 基础颜色图例
    for i, (label, color) in enumerate(color_mapping.items()):
        # 将0-255转换为0-1范围
        norm_color = tuple(c/255 for c in color)
        ax1.add_patch(Rectangle((0, i), 1, 0.8, color=norm_color))
        ax1.text(1.1, i + 0.4, f"{label}: {color}", va='center')

    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, len(color_mapping))
    ax1.set_title('基础颜色图例')
    ax1.axis('off')

    # 2. 创建混合色谱
    # 生成混合比例矩阵
    n_types = len(color_mapping)
    colors = list(color_mapping.values())

    # 创建渐变网格
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    # 计算混合颜色 (示例使用两种主色混合)
    rgb_grid = np.zeros((100, 100, 3))
    for i in range(100):
        for j in range(100):
            ratio1 = x[j]  # 第一种颜色比例
            ratio2 = 1 - x[j]  # 第二种颜色比例

            # 使用前两种颜色作为示例
            mixed = (
                int(colors[0][0]*ratio1 + colors[1][0]*ratio2),
                int(colors[0][1]*ratio1 + colors[1][1]*ratio2),
                int(colors[0][2]*ratio1 + colors[1][2]*ratio2)
            )
            rgb_grid[i, j] = [c/255 for c in mixed]

    # 绘制色谱
    im = ax2.imshow(rgb_grid, extent=[0, 1, 0, 1], origin='lower')
    ax2.set_xlabel(f'{list(color_mapping.keys())[0]} 比例')
    ax2.set_ylabel(f'{list(color_mapping.keys())[1]} 比例')
    ax2.set_title('颜色混合比例色谱')

    plt.tight_layout()
    plt.savefig('color_spectrum.png', dpi=150)
    plt.show()
    print("色谱图已保存为 color_spectrum.png")

# 使用您的颜色映射
color_mapping = {
    'JZ_Power': (255, 0, 255),     # 品红
    'SYFW_Power': (255, 0, 0),     # 红色
    'LDGC_Power': (0, 255, 0),     # 绿色
    'GY_Power': (0, 0, 255),       # 蓝色
    'GG_Power': (255, 255, 0),     # 黄色
    'DLJT_Power': (0, 255, 255)    # 青色
}

generate_color_spectrum(color_mapping)
