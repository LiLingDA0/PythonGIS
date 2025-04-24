import pandas as pd
import matplotlib.pyplot as plt
import os  # 导入 os 模块

# 获取当前项目路径
project_path = os.path.dirname(os.path.abspath(__file__))

# 定义文件路径变量
file_path = os.path.join(project_path, 'RES', 'GDD04.csv')  # 使用 os.path.join 构建路径data = pd.read_csv(file_path)

data = pd.read_csv(file_path)

factor_name = data.columns[0]

# 描述性统计
print(data[factor_name].describe())

# 直方图
plt.figure(figsize=(10, 5))
plt.hist(data[factor_name], bins=30, alpha=0.7, color='blue')
plt.title(f'Histogram of {factor_name}')
plt.xlabel(factor_name)
plt.ylabel('Frequency')
plt.show()

# 箱线图
plt.figure(figsize=(10, 5))
plt.boxplot(data[factor_name])
plt.title(f'Boxplot of {factor_name}')
plt.ylabel(factor_name)
plt.show()

# 其他分析可以根据需求添加，例如t检验等