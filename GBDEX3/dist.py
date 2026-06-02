import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

# # 读取数据
# df = pd.read_csv('data/0311_foodT_cluster_clip.csv')
#
# # 转换为弧度坐标
# coordinates = np.radians(df[['lat', 'lon']].values)
#
# # 计算最近邻距离（排除自身）
# knn = NearestNeighbors(metric='haversine', n_neighbors=2)
# knn.fit(coordinates)
# distances, _ = knn.kneighbors(coordinates)
#
# # 转换为米单位（原方案*1000）
# earth_radius_m = 6371 * 1000  # 地球半径换算为米
# df['Ndist'] = distances[:, 1] * earth_radius_m
#
# # 按分类统计平均距离（保留2位小数）
# result = df.groupby('foodT')['Ndist'].mean().round(2).reset_index()
# result.columns = ['Food Category', 'Average Nearest Distance (m)']
#
# # 生成带时间戳的报告
# timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# report = f"""食品分类最近邻距离分析报告
# 生成时间：{timestamp}
# ----------------------------------
# {result.to_string(index=False)}
#
# * 注：距离单位为米，保留两位小数"""
# with open('distance_report.txt', 'w', encoding='utf-8') as f:
#     f.write(report)
#
# df.to_parquet('data/0311_foodT_dist.parquet', engine='pyarrow')
# df.to_csv('data/0311_foodT_dist.csv', index=False)

import pandas as pd

# 读取Parquet文件
df = pd.read_parquet('data/0311_foodT_dist.parquet')

# 分组统计
result = df.groupby('foodT').agg(
    total_count=('Ndist', 'size'),
    avg_Ndist=('Ndist', 'mean'),
    max_Ndist=('Ndist', 'max'),
    count_lt_200=('Ndist', lambda x: (x < 200).sum()),
    count_lt_500=('Ndist', lambda x: (x < 500).sum()),
    count_gt_1000=('Ndist', lambda x: (x > 1000).sum()),
    count_gt_2000=('Ndist', lambda x: (x > 2000).sum())
).reset_index()

# 保存结果到Excel（可选）
result.to_csv('data/0311_foodT_dist_summary.csv', index=False)

