
import pandas as pd

# 读取CSV文件
df = pd.read_csv('data/0311_foodT_cluster_clip.csv')

# 统计cluster列的各类值个数
cluster_counts = df['cluster'].value_counts()
print("Cluster统计结果：")
print(cluster_counts)

# 统计foodT列的各类值个数
foodT_counts = df['foodT'].value_counts()
print("\nfoodT统计结果：")
print(foodT_counts)
import pandas as pd

# 读取CSV文件
df = pd.read_csv('data/0311_foodT_cluster_clip.csv')

# 统计cluster列的各类值个数
cluster_counts