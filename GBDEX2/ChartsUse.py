import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解负号显示问题

df = pd.read_parquet('data/DataFinalAnalys.parquet', engine='pyarrow')

# 数据预处理
value_counts = df['review_judege_KM_V3'].value_counts()
total = len(df)
percentages = (value_counts / total * 100).round(1)

# 创建charts文件夹
os.makedirs('charts', exist_ok=True)

# 绘制饼图
plt.figure(figsize=(10, 8))
patches, texts, autotexts = plt.pie(
    value_counts,
    labels=[f"{label}\n({count} | {percentages[label]}%)"
           for label, count in zip(value_counts.index, value_counts)],
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 10}
)

# 设置标题
plt.title(f"关注度分布饼图 (Total: {total} samples)", fontsize=14, pad=20)

# 调整布局并保存
plt.tight_layout()
plt.savefig('charts/pie_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# 预处理price_judege_KM_V3数据
price_counts = df['price_judge_KM_V3'].value_counts()

# 第一个饼图：免费与非免费统计
plt.figure(figsize=(10, 8))

# 合并非免费类别
free_vs_paid = pd.Series({
    '免费': price_counts.get('免费', 0),
    '非免费': price_counts[['便宜', '一般', '昂贵']].sum()
})

# 绘制免费/付费饼图
patches1, texts1, autotexts1 = plt.pie(
    free_vs_paid,
    labels=[f"{label}\n({count} | {(count/free_vs_paid.sum()*100):.1f}%)"
           for label, count in free_vs_paid.items()],
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 10}
)
plt.title(f"免费与非免费统计饼图 (Total: {len(df)} samples)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('charts/pie_chart_free_vs_paid.png', dpi=300, bbox_inches='tight')
plt.show()

# 第二个饼图：非免费细分统计
plt.figure(figsize=(10, 8))

# 提取非免费数据
paid_data = price_counts[['便宜', '一般', '昂贵']]
paid_total = paid_data.sum()

# 绘制非免费细分饼图
patches2, texts2, autotexts2 = plt.pie(
    paid_data,
    labels=[f"{label}\n({count} | {(count/paid_total*100):.1f}%)"
           for label, count in paid_data.items()],
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 10}
)
plt.title(f"价格细分统计饼图 (Total Paid: {paid_total} samples)",
         fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('charts/pie_chart_paid_breakdown.png', dpi=300, bbox_inches='tight')
plt.show()

