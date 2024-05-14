#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 11:20
# @Author  : quasdo
# @Site    : 
# @File    : tsne_example.py
# @Software: PyCharm

from sklearn.manifold import TSNE
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# example to use tsne
digits = datasets.load_digits(n_class=6)
X, y = digits.data, digits.target
n_samples, n_features = X.shape
print('X.shape: ')
print(X.shape)
print('y.shape: ')
print(y.shape)
print('n_samples: ')
print(n_samples)
print('n_features: ')
print(n_features)

# display the data
n = 20 # 20 digits per line, 20 digits per column
img = np.zeros((10 * n, 10 * n))
for i in range(n):
    ix = 10 * i + 1
    for j in range(n):
        iy = 10 * j + 1
        # draw the digit on the corresponding img
        img[ix: ix + 8, iy: iy + 8] = X[i * n + j].reshape((8, 8))
plt.figure(figsize=(8, 8))
plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
#plt.show()

# t-SNE, init初始化, random_state随机数种子（保证可重现性）
tsne = TSNE(n_components=2, init='pca', random_state=602)
X_tsne = tsne.fit_transform(X)
print('X_tsne.shape:')
print(X_tsne.shape)

# 8 * 8的数据降维到2
print("Org data dimension is {}. "
      "Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

# 降维后数据的最大值和最小值
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# 确保数据在相同的范围内
X_norm = (X_tsne - x_min) / (x_max - x_min)
plt.figure(figsize=(8, 8))
print('X_norm.shape: ')
print(X_norm.shape)

for i in range(X_norm.shape[0]):
    # 2维数据，用颜色标记不同类别的数据点
    # plt.cm.Set1()是Matplotlib中的一个颜色映射（colormap）
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()
