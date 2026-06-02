import geopandas as gpd
import numpy as np

# 加载 BlockPower.shp 文件
block_power_path = 'PProcess/BlockPower.shp'
block_gdf = gpd.read_file(block_power_path)

# 定义POI类型到颜色的映射
color_mapping = {
    'JZ_Power': (255, 0, 0),     # 品红-居住-红
    'SYFW_Power': (255, 0, 255),     # 红色-商业-品红
    'LDGC_Power': (0, 255, 0),     # 绿色-绿地-绿
    'GY_Power': (0, 255, 255),       # 蓝色-工业-青
    'GG_Power': (255, 255, 0),     # 黄色-公共-黄
    'DLJT_Power': (0, 0, 255)    # 青色-交通-蓝
}

# 新建 R, G, B 字段并初始化为0
block_gdf['R'] = 0
block_gdf['G'] = 0
block_gdf['B'] = 0

# 计算加权颜色值
for idx, row in block_gdf.iterrows():
    total_weight = 0
    weighted_r = 0
    weighted_g = 0
    weighted_b = 0

    for field, color in color_mapping.items():
        power = row[field]
        if power > 0:
            total_weight += power
            weighted_r += power * color[0]
            weighted_g += power * color[1]
            weighted_b += power * color[2]

    if total_weight > 0:
        block_gdf.at[idx, 'R'] = int(round(weighted_r / total_weight))
        block_gdf.at[idx, 'G'] = int(round(weighted_g / total_weight))
        block_gdf.at[idx, 'B'] = int(round(weighted_b / total_weight))

# 另存为 block_RGB.shp
output_path = 'PProcess/block_RGB_2.shp'
block_gdf.to_file(output_path)

print(f"成功保存为 {output_path}，包含 R、G、B 字段！")
