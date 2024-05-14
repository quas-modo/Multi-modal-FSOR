#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/19 17:20
# @Author  : quasdo
# @Site    : 
# @File    : visualization.py
# @Software: PyCharm

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# load the dataset
image = np.load('./data/query_image.npy').reshape(150, 21168)
target = np.load('./data/query_target.npy').reshape(150)

# t-SNE
tsne = TSNE(n_components=2, init='pca')
image_tsne = tsne.fit_transform(image)
print("Org data dimension is {}. "
      "Embedded data dimension is {}".format(image.shape[-1], image_tsne.shape[-1]))
image_min, image_max = image_tsne.min(0), image_tsne.max(0)
image_norm = (image_tsne - image_min) / (image_max - image_min)
plt.figure(figsize=(8, 8))
for i in range(image_norm.shape[0]):
    plt.text(image_norm[i, 0], image_norm[i, 1], str(target[i]), color=plt.cm.Set1(target[i]),
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()