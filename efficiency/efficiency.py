#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/20 20:50
# @Author  : quasdo
# @Site    : 
# @File    : efficiency.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np

# Sample data: method names, GFLOPs (x-values), accuracy (y-values), and categories
methods = ['MCM', 'CoOp', 'CoCoOp', 'LoCoOp', 'Dual-Adapter']
training_time  = [0, 6.2, 4.1, 3.8, 0]
fpr = [42.82, 36.83, 35.53, 33.98, 30.50]
auroc = [90.76, 91.93, 91.99, 92.69, 93.62]
categories = ['Zero-shot Methods', 'Non-prior Methods', 'Non-prior Methods', 'Non-prior Methods', 'Prior-based Methods']

# Map each category to a marker type
category_markers = {
    'Zero-shot Methods': 'o',
    'Non-prior Methods': '^',
    'Prior-based Methods': '*'
}

# Map each method to a color (optional, for better distinction)
colors = ['#6DC5D1', '#DD761C','#FEB941', '#FDE49E', '#006769']

# Create the plot
fig, ax = plt.subplots()
for i, method in enumerate(methods):
    ax.scatter(training_time[i], auroc[i], marker=category_markers[categories[i]], color=colors[i], s=170, label=categories[i] if i == 0 or categories[i] != categories[i-1] else "")
    ax.annotate(method, (training_time[i], auroc[i]), textcoords="offset points", xytext=(0, -25), ha='center',
                fontsize=12, color='black', fontfamily='sans-serif')

ax.set_xlim([-0.8, 7])
ax.set_ylim([90, 94])


# Labels and title
ax.set_xlabel('Training Time')
ax.set_ylabel('AUROC')

# Legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # removing duplicate labels
ax.legend(by_label.values(), by_label.keys())


plt.tight_layout()

# Show the plot
plt.show()
