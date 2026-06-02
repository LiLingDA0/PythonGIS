import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取SHP文件
gdf = gpd.read_file("PProcess/block_RGB_2.shp")

# 确保字段存在并转换为整数
for field in ['R', 'G', 'B']:
    if field in gdf.columns:
        gdf[field] = gdf[field].astype(int).clip(0, 255)  # 确保值在0-255范围
    else:
        raise ValueError(f"字段 '{field}' 不存在于Shapefile中")

# 创建RGB颜色列（格式: #RRGGBB）
gdf['rgb_color'] = gdf.apply(
    lambda row: f"#{row['R']:02x}{row['G']:02x}{row['B']:02x}",
    axis=1
)



# 绘制地图
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color=gdf['rgb_color'], edgecolor='black', linewidth=0.5)

# 添加图例（各个用地类型的颜色）
legend_elements = [
    mpatches.Patch(color=(255 / 255, 0 / 255, 255 / 255), label='商业'),
    mpatches.Patch(color=(255/255, 0/255, 0/255), label='居住'),
    mpatches.Patch(color=(255 / 255, 255 / 255, 0 / 255), label='公共'),
    mpatches.Patch(color=(0/255, 255/255, 0/255), label='绿地'),
    mpatches.Patch(color=(0/255, 255/255, 255/255), label='工业'),
    mpatches.Patch(color=(0/255, 0/255, 255/255), label='交通'),
    mpatches.Patch(color=(0/255, 0/255, 0/255), label='其它')
]
ax.legend(handles=legend_elements, loc='lower left', title="用地类型")

plt.title("石家庄混合功能区RGB合成分布图")
plt.xlabel("经度")
plt.ylabel("纬度")
plt.savefig('Result/RGB_Map_2.png', dpi=300)
plt.show()

# 保存高清图像
