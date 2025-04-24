import pandas as pd
import numpy as np

# 读取数据
df = pd.read_parquet('data/steam-game-store.parquet', engine='pyarrow')

# 定义通用转换函数
def clean_currency(col):
    return (
        pd.to_numeric(
            col.str.replace(r'[^\d.]', '', regex=True),
            errors='coerce'
        )
        .fillna(0)
        .astype(float)
    )

# 处理original_price
df['original_price_lubi'] = clean_currency(df['original_price'])
df = df.drop('original_price', axis=1)

# 处理discounted_price/
df['discounted_price_lubi'] = clean_currency(df['discounted_price'])
df = df.drop('discounted_price', axis=1)

# 处理discount_percentage
df['discount_percentage_%'] = (
    df['discount_percentage']
    .fillna('0')  # 先填充空值
    .str.replace(r'[^0-9]', '', regex=True)  # 严格去除非数字字符
    .replace('', '0')  # 处理空字符串
    .astype(int)
)
df = df.drop('discount_percentage', axis=1)

# 处理评价字段
review_cols = [
    'overall_review_%', 'overall_review_count',
    'recent_review_%', 'recent_review_count'
]

for col in review_cols:
    df[col] = (
        pd.to_numeric(df[col], errors='coerce')
        .fillna(0)
        .astype(int)
    )

# 异常值检测报告
print("异常值报告：")
print("1. 异常折扣百分比：")
print(df[df['discount_percentage_%'] > 100][['title', 'discount_percentage_%']])

print("\n2. 异常价格值：")
print(df[(df['original_price_lubi'] < 0) | (df['discounted_price_lubi'] < 0)])

print("\n3. 异常评分百分比：")
print(df[(df['overall_review_%'] < 0) | (df['overall_review_%'] > 100)])

# 保存处理结果
df.to_parquet('data/AllDataPreP.parquet', engine='pyarrow')

print("\n数据处理完成，已保存为 AllDataPreP.parquet")

# 保存选定列数据
selected_columns = ['app_id', 'title', 'original_price_lubi', 'overall_review_count']
df[selected_columns].to_parquet('data/selectedDataPreP.parquet', engine='pyarrow')
print("\n选定列数据已保存为 selectedDataPreP.parquet")


