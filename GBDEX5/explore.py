import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取CSV文件
file_path = 'C:\\Users\\32001\\Desktop\\work\\Python\\GBD\\GBDEX5\\Data\\hebei.csv'
data = pd.read_csv(file_path)

# 查看数据的前几行
print(data.head())
# 按六年为一组进行分组（1995-2018共24年，分为四组）
data['年份组'] = (data['年份'] - 1995) // 6

# 修改为各年份原始数据的折线图
plt.figure(figsize=(12, 6))

# 直接使用原始年份数据（取消分组）
plt.plot(data['年份'], data['二氧化碳排放总量（万吨）'],
        marker='o', linestyle='-', color='royalblue', linewidth=2)

# 设置图表元素
plt.title('各年份二氧化碳排放总量变化趋势')
plt.xlabel('年份')
plt.ylabel('排放总量（万吨）')
plt.grid(True, linestyle='--', alpha=0.5)

# 设置x轴刻度为完整年份
plt.xticks(ticks=data['年份'].unique())

# 添加数据标签
for year, value in zip(data['年份'], data['二氧化碳排放总量（万吨）']):
    plt.text(year, value+50, f'{value:.0f}',
            ha='center', va='bottom', fontsize=9)

plt.savefig('Result/yearly_co2_trend.png', dpi=300, bbox_inches='tight')
plt.close()



# 绘制箱型图
plt.figure(figsize=(12, 6))
sns.boxplot(
    x='年份组',
    y='二氧化碳排放总量（万吨）',
    data=data,
    hue='年份组',  # 新增hue映射
    palette='pastel',
    legend=False  # 禁用默认图例
)

plt.title('二氧化碳排放总量分布箱型图（六年分组）')
plt.xlabel('年份区间')
plt.ylabel('二氧化碳排放总量（万吨）')

# 生成更准确的组名标签
year_ranges = [
    f"{1995 + 6*i}-{1995 + 6*i + 5}" for i in sorted(data['年份组'].unique())
]
plt.xticks(ticks=data['年份组'].unique(), labels=year_ranges)
plt.savefig('Result/boxplot_co2_6year_groups.png', dpi=300, bbox_inches='tight')
plt.close()


# 绘制散点图来观察经济水平（GDP）与二氧化碳排放之间的关系
plt.figure(figsize=(12, 6))
plt.scatter(data['地区生产总值(亿元)'], data['二氧化碳排放总量（万吨）'], alpha=0.7, color='red', label='实际数据')

# 添加线性拟合线及参数
import numpy as np
from scipy import stats
x = data['地区生产总值(亿元)']
y = data['二氧化碳排放总量（万吨）']

# 执行线性回归
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# 计算R平方和皮尔逊相关系数
y_pred = p(x)
y_mean = np.mean(y)
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - y_mean)**2)
r_squared = 1 - (ss_res / ss_tot)
# 使用scipy计算相关系数和p值
pearson_r, p_value = stats.pearsonr(x, y)  # 替换numpy方法

# 生成参数文本（添加p值显示）
eq_text = f'y = {z[0]:.2f}x + {z[1]:.2f}'
r2_text = f'R2 = {r_squared:.3f}'
pearson_text = f'pearson_r = {pearson_r:.3f} (p={p_value:.3e})'  # 新增p值
param_text = f'{eq_text}\n{r2_text}\n{pearson_text}'


plt.plot(x, p(x), "b--", linewidth=2, label='线性拟合趋势线')

# 在右上角添加参数标注（调整文本框位置）
plt.text(0.95, 0.25, param_text,  # 调整y位置从0.15改为0.25
        transform=plt.gca().transAxes,
        ha='right', va='bottom',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.title('地区生产总值与二氧化碳排放总量的关系')
plt.xlabel('地区生产总值(亿元)')
plt.ylabel('二氧化碳排放总量（万吨）')
plt.grid(True)
plt.legend()
plt.savefig('Result/scatter_co2_emissions.png', dpi=300, bbox_inches='tight')
plt.close()




# 新增代码：计算年度增长率并绘制折线图
plt.figure(figsize=(12, 6))

# 确保按年份排序
data_sorted = data.sort_values('年份')

# 计算年度增长率（百分比）
data_sorted['增长率'] = data_sorted['二氧化碳排放总量（万吨）'].pct_change() * 100

# 绘制折线图
plt.plot(data_sorted['年份'], data_sorted['增长率'],
        marker='o', linestyle='-', color='teal', linewidth=2)

# 设置图表元素
plt.title('河北省二氧化碳排放年度增长率变化趋势')
plt.xlabel('年份')
plt.ylabel('增长率 (%)')
plt.grid(True, linestyle='--', alpha=0.7)

# 标记数据点
for year, rate in zip(data_sorted['年份'], data_sorted['增长率']):
    if not pd.isna(rate):
        plt.text(year, rate+0.2, f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=9)

plt.savefig('Result/co2_growth_rate_linechart.png', dpi=300, bbox_inches='tight')
plt.close()

# 新增代码：优化双Y轴显示
plt.figure(figsize=(14, 8))
ax = plt.gca()

# 单位转换（保持原始数据不变）
data['人均GDP_万元'] = data['人均地区生产总值(元)'] / 10000

# 左侧Y轴指标（排放相关）
ax.plot(data['年份'], data['人均二氧化碳排放量（吨/人）'],
        marker='o', linestyle='-', color='steelblue',
        linewidth=2, label='人均排放（吨/人）')
ax.plot(data['年份'], data['二氧化碳排放强度（吨/万元）'],
        marker='s', linestyle='--', color='darkorange',
        linewidth=2, label='排放强度（吨/万元）')

# 设置左侧Y轴
ax.set_xlabel('年份', fontsize=12)
ax.set_ylabel('排放指标单位', fontsize=12)
ax.tick_params(axis='y', colors='dimgray')
ax.grid(True, linestyle='--', alpha=0.7)

# 右侧Y轴（经济指标）
ax2 = ax.twinx()
ax2.plot(data['年份'], data['人均GDP_万元'],
        marker='^', linestyle=':', color='forestgreen',
        linewidth=2, label='人均GDP（万元）')

# 新增网格对齐代码
from matplotlib.ticker import MaxNLocator

# 设置统一刻度数
ax.yaxis.set_major_locator(MaxNLocator(5))
ax2.yaxis.set_major_locator(MaxNLocator(5))

# 计算比例因子
left_min, left_max = ax.get_ylim()
right_min, right_max = data['人均GDP_万元'].min()*0.9, data['人均GDP_万元'].max()*1.1
scale_factor = (right_max - right_min) / (left_max - left_min)

# 对齐刻度函数
def scale_y(lim, factor):
    return lim[0]*factor + right_min - left_min*factor, lim[1]*factor + right_min - left_min*factor

# 应用对齐
ax2.set_ylim(scale_y(ax.get_ylim(), scale_factor))

# 设置右侧Y轴样式
ax2.set_ylabel('人均GDP（万元）', fontsize=12)
ax2.tick_params(axis='y', colors='forestgreen')
ax2.grid(True, linestyle='--', alpha=0.7, color='forestgreen')

# 统一设置
plt.title('关键指标趋势分析（1995-2018）', pad=20)
plt.xticks(data['年份'].unique())

# 合并图例
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2,
         loc='upper left', frameon=True)

plt.savefig('Result/optimized_dual_axis.png', dpi=300, bbox_inches='tight')
plt.close()


