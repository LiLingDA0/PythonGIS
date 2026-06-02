import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 加载数据
output_path = 'PProcess/blockPower.shp'
block_gdf = gpd.read_file(output_path)

# 检查是否为空
if block_gdf.empty:
    raise ValueError("block_gdf 为空，请检查文件路径或内容。")

# 过滤掉空几何和无效几何
block_gdf = block_gdf[(block_gdf.geometry.notnull()) & (block_gdf.geometry.is_valid)]

# 再次检查是否为空
if block_gdf.empty:
    raise ValueError("所有几何对象已被过滤，可能是数据中存在大量空/无效几何，请检查数据完整性。")

# 获取原始坐标系下的 bounds
bounds = block_gdf.total_bounds

# 检查 bounds 是否包含 NaN 或 Inf
if np.any(np.isnan(bounds)) or np.any(np.isinf(bounds)):
    raise ValueError("block_gdf 的 total_bounds 包含 NaN 或 Inf，请检查原始数据坐标。")

# 检查范围合理性（防止 max <= min）
if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
    raise ValueError("地理范围不合理（最大坐标小于等于最小坐标），请检查数据坐标系统和内容。")

# 确保原始 CRS 存在
if block_gdf.crs is None:
    raise ValueError("block_gdf 没有定义 CRS，请指定正确的坐标系后再继续。")

# 转换为 WGS84（经纬度坐标系）用于设置范围
block_gdf_wgs84 = block_gdf.to_crs(epsg=4326)
bounds_wgs84 = block_gdf_wgs84.total_bounds



# 定义目标投影（CGCS2000_3_Degree_GK_CM_102E）
target_crs = ccrs.epsg(4542)

# 创建图形（使用 cartopy 的 GeoAxes）
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': target_crs})

# 步骤1：验证边界值有效性（在调用 set_extent 前添加）
import numpy as np

print("原始边界值:", bounds_wgs84)
# 检查边界值是否有效
if not all(np.isfinite(bounds_wgs84)):
    print(f"无效边界值: {bounds_wgs84}")
    # 步骤2：添加默认安全范围（示例值，需根据实际地图调整）
    safe_bounds = [-180, -90, 180, 90]  # 全球范围
    bounds_wgs84 = safe_bounds

ax.set_extent([0,360,40,90],crs=ccrs.PlateCarree())

#[114,115,37,39]
# 绘制 RGB 数据（使用原始坐标系）
block_gdf.plot(ax=ax, facecolor='none', edgecolor='none',
               color=[(r / 255, g / 255, b / 255) for r, g, b in zip(block_gdf['R'], block_gdf['G'], block_gdf['B'])])

# 添加标题
plt.title("石家庄用地类型RGB混合图", fontsize=16)

# 添加指北针
arrowprops = dict(facecolor='black', arrowstyle="->")
ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
            fontsize=16, ha='center', va='center', rotation=0,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"),
            arrowprops=arrowprops)

# 添加比例尺
def add_scale_bar(ax, location=(0.5, 0.05), length=10000):
    """在指定位置添加比例尺"""
    x0, y0 = location
    scalebar = Line2D([x0, x0 + length / 100000], [y0, y0], color='black', linewidth=2, transform=ax.transAxes)
    ax.add_line(scalebar)
    ax.text(x0, y0 - 0.02, f'{length/1000} km', transform=ax.transAxes, ha='center', fontsize=10)

add_scale_bar(ax)

# 添加图例（各个用地类型的颜色）
legend_elements = [
    mpatches.Patch(color=(255/255, 0/255, 255/255), label='JZ_Power'),
    mpatches.Patch(color=(255/255, 0/255, 0/255), label='SYFW_Power'),
    mpatches.Patch(color=(0/255, 255/255, 0/255), label='LDGC_Power'),
    mpatches.Patch(color=(0/255, 0/255, 255/255), label='GY_Power'),
    mpatches.Patch(color=(255/255, 255/255, 0/255), label='GG_Power'),
    mpatches.Patch(color=(0/255, 255/255, 255/255), label='DLJT_Power')
]
ax.legend(handles=legend_elements, loc='lower left', title="用地类型")

# 添加制图人信息
ax.text(0.01, 0.01, '制图人：张三', transform=ax.transAxes, fontsize=10, verticalalignment='bottom')

# 显示地图
plt.tight_layout()
plt.show()

# 保存图片
fig.savefig('Shijiazhuang_RGB_Map.png', dpi=300, bbox_inches='tight')
print("地图已保存为 Shijiazhuang_RGB_Map.png")
