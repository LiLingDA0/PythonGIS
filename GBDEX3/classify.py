import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 读取数据
df = pd.read_parquet('data/0311_foodT.parquet')

# 2. 提取特征并标准化
coords = df[['lat', 'lon']]
scaler = StandardScaler()
scaled_coords = scaler.fit_transform(coords)

# 3. 计算不同K值的SSE（肘部法则）
sse = []
k_range = range(2, 11)  # 通常测试2-10个聚类

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(scaled_coords)
    sse.append(kmeans.inertia_)

# 4. 绘制肘部图
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.savefig('elbow_plot.png')  # 保存图片
plt.show()

# 5. 选择最佳K值后进行聚类（假设选择K=5）
best_k = 4  # 根据肘部图拐点手动设置
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
df['cluster'] = final_kmeans.fit_predict(scaled_coords)

plt.scatter(df['lon'], df['lat'], c=df['cluster'], cmap='tab20', s=10)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()



