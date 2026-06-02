import pandas as pd

# 读取Reclass.csv文件
reclass_df = pd.read_csv('data/Reclass.csv')

# 创建映射字典：原始类别 -> (大类, 影响度)
mapping_dict = dict(zip(reclass_df['原始类别'], zip(reclass_df['大类'], reclass_df['影响度'].astype(int))))

# 读取Parquet文件
parquet_df = pd.read_parquet('data/0311.parquet')

# 映射新字段DaLei和Power
def map_category(category):
    result = mapping_dict.get(category, ('未知', 0))
    return pd.Series({
        'DaLei': result[0],
        'Power': int(result[1])  # 强制转为整型
    })

# 应用映射
parquet_df[['DaLei', 'Power']] = parquet_df['category_sec'].apply(map_category)

# 确保Power是整型
parquet_df['Power'] = parquet_df['Power'].fillna(0).astype(int)

# 保存结果
parquet_df.to_parquet('0311_with_new_fields.parquet', index=False)

