import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os

KernelMethod = 'poly'  # 'rbf'、'sigmoid'、'linear'、'poly'

# 设置全局字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 替换为支持Unicode的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
data = pd.read_parquet('Data/SJZAirQuality.parquet')

# 定义自变量和因变量
X = data[['24hPM10avg', '24hSO2avg', '24hNO2avg', '24hCOavg', '24hO3avg']]
y = data['PM25Class']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据集为训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 使用SMOTE处理类别不平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 定义不同的核函数
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# 存储每个核函数下的性能指标
accuracies_kernel = []
f1_scores_kernel = []
precisions_kernel = []
recalls_kernel = []

# 固定gamma值进行测试
fixed_gamma = 0.1

# 遍历每个核函数
for kernel in kernels:
    print(f"Processing kernel: {kernel}")
    # 创建SVM模型
    svm_model = SVC(kernel=kernel, C=1.0, gamma=fixed_gamma, probability=True)
    svm_model = OneVsRestClassifier(svm_model)

    # 训练模型
    svm_model.fit(X_resampled, y_resampled)

    # 进行预测
    y_pred = svm_model.predict(X_test)

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)

    # 存储性能指标
    accuracies_kernel.append(accuracy)
    f1_scores_kernel.append(f1_macro)
    precisions_kernel.append(precision_macro)
    recalls_kernel.append(recall_macro)

# 绘制图表并保存
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
plt.bar(kernels, accuracies_kernel)
plt.ylim(0.8, 1)  # 设置y轴范围
plt.xlabel('Kernel')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Kernel')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.bar(kernels, f1_scores_kernel)
plt.ylim(0.8, 1)  # 设置y轴范围
plt.xlabel('Kernel')
plt.ylabel('F1 Score (Macro)')
plt.title('F1 Score vs Kernel')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.bar(kernels, precisions_kernel)
plt.ylim(0.8, 1)  # 设置y轴范围
plt.xlabel('Kernel')
plt.ylabel('Precision (Macro)')
plt.title('Precision vs Kernel')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.bar(kernels, recalls_kernel)
plt.ylim(0.8, 1)  # 设置y轴范围
plt.xlabel('Kernel')
plt.ylabel('Recall (Macro)')
plt.title('Recall vs Kernel')
plt.grid(True)

plt.tight_layout()
plt.savefig('Results/SVM/Kernel_Comparison.png')
plt.close()
