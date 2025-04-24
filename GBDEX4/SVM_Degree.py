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

# 定义不同的degree值
degree_values = [2, 3, 4, 5]  # 多项式核函数的阶数

# 存储每个degree值下的性能指标
accuracies_degree = []
f1_scores_degree = []
precisions_degree = []
recalls_degree = []

# 遍历每个degree值
for degree in degree_values:
    # 创建SVM模型，指定kernel='poly'和degree
    svm_model = SVC(kernel='poly', C=1.0, gamma='scale', degree=degree, probability=True)
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
    accuracies_degree.append(accuracy)
    f1_scores_degree.append(f1_macro)
    precisions_degree.append(precision_macro)
    recalls_degree.append(recall_macro)

# 绘制图表并保存
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
plt.plot(degree_values, accuracies_degree, marker='o')
plt.xlabel('Degree')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Degree')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(degree_values, f1_scores_degree, marker='o')
plt.xlabel('Degree')
plt.ylabel('F1 Score (Macro)')
plt.title('F1 Score vs Degree')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(degree_values, precisions_degree, marker='o')
plt.xlabel('Degree')
plt.ylabel('Precision (Macro)')
plt.title('Precision vs Degree')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(degree_values, recalls_degree, marker='o')
plt.xlabel('Degree')
plt.ylabel('Recall (Macro)')
plt.title('Recall vs Degree')
plt.grid(True)

plt.tight_layout()
plt.savefig('Results/SVM/poly_Degree.png')
plt.close()

