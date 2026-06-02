import numpy as np

# ========== 矩阵创建 ==========
# 从列表创建
A = np.array([[1,2], [3,4]])  # 2x2矩阵

# 特殊矩阵
zeros = np.zeros((3,2))       # 3x2零矩阵
ones = np.ones((2,3))         # 2x3单位矩阵
identity = np.eye(3)          # 3x3单位阵
random_mat = np.random.rand(2,2) # 2x2随机矩阵

# ========== 矩阵索引 ==========
print(A[0,1])     # 输出 2（第0行第1列）
print(A[:,1])     # 输出 [2,4]（第1列）
print(A[1,:])     # 输出 [3,4]（第1行）
print(A[A > 2])   # 输出 [3,4]（布尔索引）

# ========== 矩阵变形 ==========
B = A.reshape(4,1)  # 变形为4x1矩阵（不修改原数据）
C = np.resize(A, (3,3))  # 自动填充重复元素

# 展平操作
flattened = A.flatten()  # [1,2,3,4]（返回拷贝）
raveled = A.ravel()      # 原数据视图

# ========== 元素级乘法 ==========
D = A * A          # 对应元素相乘
E = np.multiply(A, A)

# ========== 矩阵点积 ==========
F = np.dot(A, A)   # 矩阵乘法
G = A @ A          # Python 3.5+ 运算符

# ========== 矩阵的迹 ==========
trace = np.trace(A)  # 1+4=5

# ========== L2范数 ==========
l2_norm = np.linalg.norm(A)  # sqrt(1+4+9+16)=5.477

# ========== 广播机制 ==========
vec = np.array([10, 20])
H = A + vec  # 自动广播为：
# [[1+10, 2+20],
#  [3+10, 4+20]]
