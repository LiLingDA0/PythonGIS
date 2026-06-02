import pandas as pd

# 读取CSV文件
file_path = 'Data/hebei.csv'
df = pd.read_csv(file_path)

# 提取省份为河北省的数据
hebei_df = df[df['省份'] == '河北省']

# 保存为Parquet格式
output_path = 'Data\HebeiCO2.parquet'
hebei_df.to_parquet(output_path, engine='pyarrow')
