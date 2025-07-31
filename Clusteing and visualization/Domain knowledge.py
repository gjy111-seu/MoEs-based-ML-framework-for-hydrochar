import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

# 读取数据
df = pd.read_excel("Data file path", sheet_name='Sheet1')


features = ['C', 'H', 'O', 'N', 'FC', 'VM', 'A']
X = df[features]

tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_embedded = tsne.fit_transform(X)


df['TSNE_1'] = X_embedded[:, 0]
df['TSNE_2'] = X_embedded[:, 1]
tsne_df = df[['TSNE_1', 'TSNE_2', 'Cluster'] + features]

