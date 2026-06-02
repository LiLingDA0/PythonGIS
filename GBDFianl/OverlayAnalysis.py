import geopandas as gpd
import os
from tqdm import tqdm

# 定义路径
input_folder = 'FinalD'
output_folder = 'PProcess'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 1. 加载POI.shp并根据Power字段生成缓冲区
print("Step 1: Generating buffer for POI based on Power field...")
poi_path = os.path.join(input_folder, 'POI.shp')
poi_gdf = gpd.read_file(poi_path)
poi_gdf['buffer'] = poi_gdf.buffer(poi_gdf['Power'])
poi_bff_gdf = gpd.GeoDataFrame(poi_gdf[['DaLei', 'Power','buffer','category_s','poiid']], geometry='buffer', crs=poi_gdf.crs)
poi_bff_path = os.path.join(output_folder, 'POI_Bff.shp')
poi_bff_gdf.to_file(poi_bff_path)
print(f"Saved POI_Bff.shp to {output_folder}")

# 2. 加载Blocks.shp，重命名并添加影响力字段
print("Step 2: Creating BlockPower.shp and adding influence fields...")
block_path = os.path.join(input_folder, 'Blocks.shp')
block_gdf = gpd.read_file(block_path)

# 新建字段
power_fields = ['JZ_Power', 'SYFW_Power', 'LDGC_Power',
                'GY_Power', 'GG_Power', 'DLJT_Power']
for field in power_fields:
    block_gdf[field] = 0

block_power_path = os.path.join(output_folder, 'BlockPower.shp')
block_gdf.to_file(block_power_path)
print(f"Saved BlockPower.shp to {output_folder}")

# 3. 计算每个面受POI各大类的影响程度
print("Step 3: Calculating influence from POI_Bff on each Block...")

# 将POI缓冲区按DaLei分类
poi_bff_dict = {}
for category in poi_bff_gdf['DaLei'].unique():
    if category == '未知':
        continue
    poi_bff_dict[category] = poi_bff_gdf[poi_bff_gdf['DaLei'] == category]

# 构建空间索引以加速相交判断
block_gdf.sindex

# 映射大类到字段名前缀
category_to_field = {
    '居住用地': 'JZ_Power',
    '商业服务业设施用地': 'SYFW_Power',
    '绿地与广场用地': 'LDGC_Power',
    '工业用地': 'GY_Power',
    '公共管理与公共服务设施用地': 'GG_Power',
    '道路与交通设施用地': 'DLJT_Power'
}

# 遍历每个面要素进行计算
with tqdm(total=len(block_gdf), desc="Processing Blocks") as pbar:
    for idx, block_row in block_gdf.iterrows():
        block_geom = block_row.geometry
        for category, bff_gdf in poi_bff_dict.items():
            intersecting_pois = bff_gdf[bff_gdf.intersects(block_geom)]
            total_power = intersecting_pois['Power'].sum()
            field_name = category_to_field.get(category, None)
            if field_name:
                block_gdf.at[idx, field_name] = total_power
        pbar.update(1)

# 保存更新后的BlockPower.shp
block_gdf.to_file(block_power_path)
print(f"Updated BlockPower.shp saved with influence values.")

# 4. 确定每个面的第一和第二影响力来源
print("Step 4: Determining top two influencing categories...")

# 新增字段
block_gdf['FstP'] = ''       # 字符串
block_gdf['SecP'] = ''
block_gdf['FstP_P'] = 0      # 整数百分比
block_gdf['SecP_P'] = 0

# 映射字段前缀到中文大类名称
field_to_category = {
    'JZ_Power': '居住用地',
    'SYFW_Power': '商业服务业设施用地',
    'LDGC_Power': '绿地与广场用地',
    'GY_Power': '工业用地',
    'GG_Power': '公共管理与公共服务设施用地',
    'DLJT_Power': '道路与交通设施用地'
}

def get_top_two(row):
    values = [(field, row[field]) for field in power_fields if row[field] > 0]
    if not values:
        return ('', '', 0, 0)  # 返回整型 0 表示空值
    elif len(values) == 1:
        category = field_to_category.get(values[0][0], '')
        return (category, '', int(round(values[0][1]/values[0][1] * 100)), 0)
    else:
        sorted_values = sorted(values, key=lambda x: x[1], reverse=True)
        total = sum(v[1] for v in sorted_values)
        fst = sorted_values[0]
        snd = sorted_values[1]
        fst_category = field_to_category.get(fst[0], '')
        snd_category = field_to_category.get(snd[0], '')
        return (
            fst_category,
            snd_category,
            int(round(fst[1] / total * 100)),
            int(round(snd[1] / total * 100))
        )


with tqdm(total=len(block_gdf), desc="Calculating Top Influences") as pbar:
    for idx, row in block_gdf.iterrows():
        fst, snd, p1, p2 = get_top_two(row)
        block_gdf.at[idx, 'FstP'] = fst
        block_gdf.at[idx, 'SecP'] = snd
        block_gdf.at[idx, 'FstP_P'] = p1
        block_gdf.at[idx, 'SecP_P'] = p2
        pbar.update(1)

# 最终保存
block_gdf.to_file(block_power_path)
print(f"Final BlockPower.shp saved with top influence information.")
