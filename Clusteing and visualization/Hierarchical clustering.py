import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE


df = pd.read_excel("Data file path", sheet_name='Sheet1')
features = ['C', 'H', 'O', 'N', 'FC', 'VM', 'A', 'T', 'RT', 'SL']
X = df[features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


linked = linkage(X_scaled, method='ward')


k_range = range(1, 11)
metrics = {
    'Silhouette': [],

}

for k in k_range:

    labels = fcluster(linked, k, criterion='maxclust')

    sse = 0
    for cluster_id in np.unique(labels):

        cluster_data = X_scaled[labels == cluster_id]

        cluster_center = cluster_data.mean(axis=0)

    metrics['Silhouette'].append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 2)
plt.plot(k_range, metrics['Silhouette'], 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')

plt.tight_layout()
plt.show()

best_k = k_range[np.nanargmax(metrics['Silhouette'])]
print(f"Recommended number of clusters: {best_k}")

df['Cluster'] = fcluster(linked, 4, criterion='maxclust')


plt.figure(figsize=(10, 6))
dendrogram(linked, labels=df.index, leaf_rotation=90, leaf_font_size=8)
plt.title("Hierarchical clustering tree")
plt.xlabel("data index")
plt.ylabel("distance")
plt.show()



X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
df['TSNE_1'], df['TSNE_2'] = X_embedded[:, 0], X_embedded[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='TSNE_1', y='TSNE_2', hue='Cluster', palette='tab10', data=df, s=70, alpha=0.8, edgecolors='k')
plt.title("t-SNE Hierarchical clustering")
plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")
plt.legend(title="Cluster")
plt.show()


