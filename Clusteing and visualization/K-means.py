import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# 读取数据
df = pd.read_excel("Data file path", sheet_name='Sheet1')


features = ['C', 'H', 'O', 'N', 'FC', 'VM', 'A']
X = df[features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_range = range(2, 11)
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=123, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette_avg)

best_k = k_range[np.argmax(silhouette_scores)]



kmeans = KMeans(n_clusters=best_k, random_state=123, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)


tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_embedded = tsne.fit_transform(X_scaled)


df['TSNE_1'] = X_embedded[:, 0]
df['TSNE_2'] = X_embedded[:, 1]

tsne_df = df[['TSNE_1', 'TSNE_2', 'Cluster'] + features]
metrics_df = pd.DataFrame({
    'Number_of_Clusters': list(k_range),
    'Silhouette_Score':  silhouette_scores
})

plt.figure(figsize=(8, 6))
sns.scatterplot(x='TSNE_1', y='TSNE_2', hue='Cluster', palette='tab10', data=df, s=70, alpha=0.8, edgecolors='k')
plt.title("t-SNE 降维可视化 - K-Means 聚类结果")
plt.xlabel("t-SNE 维度 1")
plt.ylabel("t-SNE 维度 2")
plt.legend(title="簇编号")
plt.show()
