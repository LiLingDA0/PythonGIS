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

KernelMethod = 'rbf'  # 'rbf'、'sigmoid'、'linear'、'poly'

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

# 定义不同的C值
C_values = np.logspace(-3, 3, 7)

# 存储每个C值下的性能指标
accuracies = []
f1_scores = []
precisions = []
recalls = []

# 遍历每个C值
for C in C_values:
    # 创建SVM模型
    svm_model = SVC(kernel=KernelMethod, C=C, probability=True)
    svm_model = OneVsRestClassifier(svm_model)

    # 训练模型
    svm_model.fit(X_resampled, y_resampled)

    # 进行预测
    y_pred = svm_model.predict(X_test)

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')

    # 存储性能指标
    accuracies.append(accuracy)
    f1_scores.append(f1_macro)
    precisions.append(precision_macro)
    recalls.append(recall_macro)

# 确保保存目录存在
os.makedirs('Results/SVM', exist_ok=True)

# 绘制图表并保存
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
plt.plot(C_values, accuracies, marker='o')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Accuracy vs C')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(C_values, f1_scores, marker='o')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('F1 Score (Macro)')
plt.title('F1 Score vs C')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(C_values, precisions, marker='o')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Precision (Macro)')
plt.title('Precision vs C')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(C_values, recalls, marker='o')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Recall (Macro)')
plt.title('Recall vs C')
plt.grid(True)
plt.savefig('Results/SVM/'+KernelMethod+'_C.png')

plt.tight_layout()
plt.close()

