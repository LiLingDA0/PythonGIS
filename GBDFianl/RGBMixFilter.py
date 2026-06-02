import geopandas as gpd

# 读取 block_RGB.shp 文件
input_path = 'PProcess/block_RGB.shp'
block_gdf = gpd.read_file(input_path)

# 确保存在 FstP_P 字段
if 'FstP_P' not in block_gdf.columns:
    raise KeyError("Shapefile 中未找到 'FstP_P' 字段，请检查数据")

# 修改满足条件的要素 RGB 值
for idx, row in block_gdf.iterrows():
    if row['FstP_P'] >= 50:
        block_gdf.at[idx, 'R'] = 0
        block_gdf.at[idx, 'G'] = 0
        block_gdf.at[idx, 'B'] = 0

# 另存为 block_RGB_2.shp
output_path = 'PProcess/block_RGB_2.shp'
block_gdf.to_file(output_path)

print(f"成功保存为 {output_path}，已更新 RGB 字段！")
