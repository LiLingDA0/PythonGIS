import numpy as np
import matplotlib.pyplot as plt
from classify import getDistortions
import os  # 导入 os 模块
import rasterio

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

K_range = range(1, 11)
distortions = getDistortions(Elements, K_range)


# 绘制肘部图
plt.plot(K_range, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
output_dir = os.path.join(project_path, 'RESULT')  # 创建结果目录
plt.savefig(os.path.join(output_dir, 'elbow_chart.png'), bbox_inches='tight', dpi=300, format='png')
plt.show()

