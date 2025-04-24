import numpy as np

# 样本数据 (4个样本，2个特征)
X_features = np.array([
    [1, 2],   # 样本1的两个特征
    [2, 6],   # 样本2
    [3, 4],   # 样本3
    [4, 5]    # 样本4
])
Y = np.array([[3], [5], [7], [9]])

# 构造设计矩阵（添加截距列）
X = np.c_[np.ones((4, 1)), X_features]

# 计算系数矩阵

XTX = X.T @ X
X_inv=np.zeros(Y.T.shape)
if np.linalg.matrix_rank(XTX) < X.shape[1]:
    print("检测到矩阵不满秩，使用伪逆计算")
    #X_inv = np.linalg.pinv(X)
else:
    X_inv = np.linalg.inv(XTX) @ X.T
Beta = X_inv @ Y



if Beta is not None:
    print(Beta)
