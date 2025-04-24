import Filter
import rasterio
import matplotlib.pyplot as plt
import os  # 导入 os 模块

# 配置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解负号显示问题

# 获取当前项目路径
project_path = os.path.dirname(os.path.abspath(__file__))

# 定义文件路径变量
file_path = os.path.join(project_path, 'RES', 'test-data')  # 使用 os.path.join 构建路径

# 打开文件并读取数据
with rasterio.open(file_path) as src:  # 指定 nodata 值
    band1 = src.read(1)  # 读取第1波段
    band5 = src.read(5)


# 对 band1 和 band5 进行中值滤波
band1_mean = Filter.f_mode(band1, 'mean', size=3)  # size 可以根据需要调整
band1_roberts = Filter.f_mode(band1_mean, 'roberts', stretch=50)
band5_median = Filter.f_mode(band5, 'median', size=3)  # size 可以根据需要调整
band5_sobel= Filter.sobel_filter(band5_median)


fig, axes = plt.subplots(2, 3)  # 创建2行3列布局
axes[0,0].imshow(Filter.normalize(band5), cmap='gray')  # 在第一个子图显示
axes[0,0].set_title('原始图像')
axes[0,1].imshow(Filter.normalize(band5_median), cmap='gray')  # 在第二个子图显示
axes[0,1].set_title('中值滤波')
axes[0,2].imshow(Filter.normalize(band5_sobel), cmap='gray')
axes[0,2].set_title('Sobel滤波')
axes[1,0].imshow(Filter.normalize(band1), cmap='gray')
axes[1,0].set_title('原始图像')
axes[1,1].imshow(Filter.normalize(band1_mean), cmap='gray')
axes[1,1].set_title('均值滤波')
axes[1,2].imshow(Filter.normalize(band1_roberts), cmap='gray')
axes[1,2].set_title('Roberts滤波')
plt.tight_layout()
plt.show()