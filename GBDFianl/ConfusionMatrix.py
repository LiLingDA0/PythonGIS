import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
file_path = r'C:\Users\32001\Desktop\work\Python\GBD\GBDFianl\FinalD\check.csv'
df = pd.read_csv(file_path)

mapping = {
    "居住用地": "居住用地",
    "商业服务业设施用地": "商业服务",
    "公共管理与公共服务设施用地": "公共设施",
    "工业用地": "工业用地",
    "绿地与广场用地": "绿地广场",
    "道路与交通设施用地": "道路交通"
}

# 应用映射并转为字符串类型
df['GHT_V_CN'] = df['GHT_V_CN'].map(mapping).astype(str)
df['FstP'] = df['FstP'].map(mapping).astype(str)
df['SecP'] = df['SecP'].map(mapping).astype(str)

# 构建预测结果
def get_prediction(row):
    if row['FstP'] == row['GHT_V_CN']:
        return row['FstP']
    elif row['SecP'] == row['GHT_V_CN']:
        return row['SecP']
    else:
        return row['FstP']
y_true = df['GHT_V_CN']
y_pred = df.apply(get_prediction, axis=1)

# 统一标签
labels = list(mapping.values())


# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred, labels=labels)

# 计算总样本数
total_samples = cm.sum()

# 计算每类的精确率（Precision = TP / (TP + FP)）
precision_per_class = cm.diagonal() / cm.sum(axis=0)

# 转换为字典用于展示
precision_dict = {label: f"{precision:.2%}" for label, precision in zip(labels, precision_per_class)}

# 绘制热图
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

# 添加标注信息
info_text = (
    f"总样本数: {total_samples}\n"
    "分类精度（查准率）:\n" +
    "\n".join([f"{k}: {v}" for k, v in precision_dict.items()])
)

# 在图中添加文本注释
plt.text(
    x=0.02, y=0.98, s=info_text,
    fontsize=10, transform=plt.gca().transAxes,
    verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1)
)

# 设置标题和坐标轴
plt.title('混淆矩阵热图')
plt.xlabel('预测分类')
plt.ylabel('实际分类')
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd

# 将混淆矩阵转为 DataFrame
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# 保存为 CSV 文件
output_file = r'C:\Users\32001\Desktop\work\Python\GBD\GBDFianl\FinalD\confusion_matrix.csv'
cm_df.to_csv(output_file, index=True, header=True)

print(f"混淆矩阵已保存至: {output_file}")

