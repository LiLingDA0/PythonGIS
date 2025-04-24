import pandas as pd
import numpy as np

# def haversine_vectorized(lon1, lat1, lon2, lat2):
#     """
#     计算经纬度距离（单位：公里）
#     参数说明：
#     lon1, lat1: 单个基准点的经纬度
#     lon2, lat2: 数组形式的待计算点经纬度
#     """
#     # 转换为弧度
#     lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
#
#     # 差值计算
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#
#     # Haversine公式
#     a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
#     c = 2 * np.arcsin(np.sqrt(a))
#     return c * 6371  # 地球平均半径
#
# # 读取数据
# house = pd.read_csv("data/0311_house_clip.csv")
# foodT = pd.read_csv("data/0311_foodT_cluster_clip.csv")
#
# # 预转换foodT坐标
# foodT_lons = foodT["lon"].values
# foodT_lats = foodT["lat"].values
#
# # 定义统计函数
# def calculate_counts(row):
#     distances = haversine_vectorized(row["lon"], row["lat"], foodT_lons, foodT_lats)
#     return pd.Series([
#         np.sum(distances <= 1),
#         np.sum(distances <= 2),
#         np.sum(distances <= 3)
#     ], index=["C1km", "C2km", "C3km"])
#
# # 应用计算
# house[["C1km", "C2km", "C3km"]] = house.apply(calculate_counts, axis=1)
#
# # 保存结果
# house.to_parquet("data/house_with_foodT_counts.parquet", engine='pyarrow')
# house.to_csv("data/house_with_foodT_counts.csv", index=False)