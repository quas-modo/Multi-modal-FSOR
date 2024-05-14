import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# import random
# labels = [random.randint(0, 1) for _ in range(1000)]

data_path = './data_style.csv'
df = pd.read_csv(data_path)

# 使用PCA进行降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.values)

label_path = './label.csv'
labels = pd.read_csv(label_path)

# 使用K均值聚类
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_pca)
# print(labels)
# labels = kmeans.labels_
# print(labels)

# 可视化结果
plt.scatter(X_pca[:, 0], X_pca[:, 1],  c=labels.values, cmap='viridis')
plt.title('PCA and K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()