import pandas as pd
import matplotlib.pyplot as plt
import os


df = pd.read_parquet('data/RGoodPercent.parquet', engine='pyarrow')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解负号显示问题

# 在现有代码后追加以下内容
plt.figure(figsize=(10, 6))
value_counts = df['overall_review_%'].value_counts().sort_index()
value_counts.plot(kind='bar', color='skyblue')
plt.title('各好评率个数统计图')
plt.xlabel('好评率')
plt.ylabel('个数')

# 确保charts目录存在
os.makedirs('charts', exist_ok=True)

# 保存图片
plt.savefig('charts/overall_review_distribution.png',
            bbox_inches='tight',
            dpi=300)
plt.show()
