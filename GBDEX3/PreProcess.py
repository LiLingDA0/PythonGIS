import pandas as pd
from foodmap import MAPPING  # 确保foodmap.py在可导入路径

# 读取数据
df = pd.read_parquet('data/0311_filter.parquet')

# 筛选餐饮类数据
df = df[df['category_sec'] == '住宅区']

# 保存结果（可调整保存路径）
df.to_parquet('data/0311_house.parquet', engine='pyarrow')

