import pandas as pd
import numpy as np

# 1. 创建示例数据集
data = {
    'Date': pd.date_range('2023-01-01', periods=5),
    'Product': ['A', 'B', 'A', 'C', 'B'],
    'Sales': [250, 310, np.nan, 400, 380],
    'Region': ['North', 'South', 'East', 'West', 'North']
}
df = pd.DataFrame(data)

# 2. 数据预览
print("原始数据：")
print(df.head())
print("\n数据结构：")
print(df.info())

# 3. 数据清洗
# 处理缺失值
df['Sales'] = df['Sales'].fillna(df['Sales'].mean()).astype(int)
df['Product'] = df['Product'].astype('category')

# 4. 数据分析
# 按产品分组统计
# 保持当前行为（未来可能需手动调整）
product_stats = df.groupby('Product', observed=False)['Sales'].agg(['sum', 'mean'])

print("\n产品统计：")
print(product_stats)

# 5. 数据合并演示
price_data = pd.DataFrame({
    'Product': ['A', 'B', 'C'],
    'UnitPrice': [50, 60, 70]
})
merged_df = pd.merge(df, price_data, on='Product')

# 6. 时间序列处理
merged_df['Month'] = merged_df['Date'].dt.month_name()

# 7. 结果保存
merged_df.to_csv('processed_sales_data.csv', index=False)
print("\n处理后的数据已保存为 processed_sales_data.csv")

# 8. 最终数据展示
print("\n最终处理结果：")
print(merged_df)


