# Relation.py
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 字段名映射字典
field_to_category = {
    'JZ_Power': '居住',
    'SYFW_Power': '商业',
    'LDGC_Power': '绿地',
    'GY_Power': '工业',
    'GG_Power': '公共',
    'DLJT_Power': '交通'
}

# ... [其他代码保持不变] ...

# 加载block_gdf数据
input_folder = 'PProcess'  # 请替换为实际路径
block_gdf = gpd.read_file(os.path.join(input_folder, 'BlockPower.shp'))  # 假设使用Shapefile格式

# 3. 影响力字段相关性分析
print("\nCalculating correlations between influence fields...")
power_fields = ['JZ_Power', 'SYFW_Power', 'LDGC_Power',
                'GY_Power', 'GG_Power', 'DLJT_Power']

# 计算皮尔逊相关系数矩阵
correlation_matrix = block_gdf[power_fields].corr()

# 重命名为中文
correlation_matrix = correlation_matrix.rename(
    columns=field_to_category,
    index=field_to_category
)

# 打印相关系数矩阵
print("\nCorrelation Matrix:")
print(correlation_matrix)

# 4. 可视化相关性
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('影响力相关性矩阵')  # 更新标题
plt.tight_layout()
plt.savefig(os.path.join('Result', 'influence_correlations.png'))
plt.show()