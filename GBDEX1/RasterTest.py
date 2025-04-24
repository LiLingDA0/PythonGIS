import rasterio
import matplotlib.pyplot as plt
import os  # 导入 os 模块
import numpy as np  # 导入 numpy 模块

# 配置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解负号显示问题

# 获取当前项目路径
project_path = os.path.dirname(os.path.abspath(__file__))

# 定义文件路径变量
file_path = os.path.join(project_path, 'RES', 'DEM.tif')  # 使用 os.path.join 构建路径

# 打开文件并读取数据
with rasterio.open(file_path) as src:  # 指定 nodata 值
    band = src.read(1)  # 读取第1波段

# 获取 band 中第二大的值且不重复
unique_values = np.unique(band)
sorted_unique_values = np.sort(unique_values)
second_largest_value = sorted_unique_values[-2] if len(sorted_unique_values) > 1 else None
print("Second largest unique value in band:", second_largest_value)

# 显示图像
plt.figure(figsize=(10, 8))
plt.imshow(band, cmap='gray', vmin=band.min(), vmax=band.max())  # 使用 viridis 颜色映射，并设置显示范围
plt.colorbar(label='像素值')
plt.title('单波段灰度图')
plt.xlabel('X 坐标')
plt.ylabel('Y 坐标')
plt.show()