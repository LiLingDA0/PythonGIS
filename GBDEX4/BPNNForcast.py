# 导入必要的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier  # 替换为MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE  # 引入SMOTE处理类别不平衡
import joblib
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 'Microsoft YaHei'
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

# 创建BPNN模型
bpnn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)  # 创建BPNN模型

bpnn_model.fit(X_resampled, y_resampled)

# 进行预测
y_pred = bpnn_model.predict(X_test)
y_pred_proba = bpnn_model.predict_proba(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
class_report = classification_report(y_test, y_pred)

print("BPNN 模型评估结果:")
print("\n")
print("Accuracy:", accuracy)
print("\n")
print("F1 Score (Macro):", f1_macro)
print("\n")
print("Classification Report:\n", class_report)
print("\n")

# 获取当前日期
current_date = datetime.now().strftime('%Y%m%d')

# 定义方法名和文件名
method_name = 'BPNN'
result_dir = f'Results/{method_name}_{current_date}'
scaler_filename = f'{result_dir}/scaler.joblib'
model_filename = f'{result_dir}/model.joblib'
report_filename = f'{result_dir}/report.txt'
confusion_matrix_filename = f'{result_dir}/confusion_matrix.png'
roc_curve_filename = f'{result_dir}/roc_curve.png'
pr_curve_filename = f'{result_dir}/pr_curve.png'

# 创建结果目录（如果不存在）
os.makedirs(result_dir, exist_ok=True)

# 保存模型
joblib.dump(scaler, scaler_filename)
joblib.dump(bpnn_model, model_filename)

# 生成混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 保存分类报告
with open(report_filename, 'w') as report_file:
    report_file.write("BPNN 模型评估结果:\n")
    report_file.write("\n")
    report_file.write(f"Accuracy: {accuracy}\n")
    report_file.write("\n")
    report_file.write(f"F1 Score (Macro): {f1_macro}\n")
    report_file.write("\n")
    report_file.write("Classification Report:\n")
    report_file.write(class_report)
    report_file.write("\nConfusion Matrix:\n")
    for row in conf_matrix:
        report_file.write(" ".join(map(str, row)) + "\n")

# 设置全局字体大小
plt.rcParams.update({'font.size': 12})

# 可视化混淆矩阵为表格
conf_matrix_df = pd.DataFrame(conf_matrix, index=data['PM25Class'].unique(), columns=data['PM25Class'].unique())

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5, linecolor='black', annot_kws={"size": 16})
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title('BPNN Confusion Matrix', fontsize=16)
plt.savefig(confusion_matrix_filename)
plt.close()

print(f"模型已保存至 {scaler_filename} 和 {model_filename}")
print(f"分类报告已保存至 {report_filename}")
print(f"混淆矩阵已保存至 {confusion_matrix_filename}")

# 绘制ROC曲线
y_test_binarized = label_binarize(y_test, classes=data['PM25Class'].unique())
n_classes = y_test_binarized.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    roc_auc[i] = roc_auc_score(y_test_binarized[:, i], y_pred_proba[:, i])

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {data["PM25Class"].unique()[i]} (area = {roc_auc[i]:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('BPNN Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(roc_curve_filename)
plt.close()

print(f"ROC曲线已保存至 {roc_curve_filename}")

# 绘制PR曲线
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    average_precision[i] = average_precision_score(y_test_binarized[:, i], y_pred_proba[:, i])

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(recall[i], precision[i], label=f'Class {data["PM25Class"].unique()[i]} (area = {average_precision[i]:.4f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('BPNN Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig(pr_curve_filename)
plt.close()

print(f"PR曲线已保存至 {pr_curve_filename}")
