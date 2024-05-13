import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(dpi=500)
x = [0, 1, 2, 4, 8, 16]

CoOp = [50.64, 44.92, 43.74, 43.29, 39.33]
LoCoOp = [49.25, 45.07, 44.15, 42.49, 40.23]
Dual = [39, 38.76, 39.24, 38.78, 38.72]
MCM = [59.91]

#tipkclx = [65.3, 66.7, 67.3, 69.5, 71]
#tipkcl = [67.27, 68.19, 69.31, 70.46, 71.81]
ours = []
color_list = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'gray']
marker_list = ['o', 'x', '+', '*', '^', 'v', '<', '>', 'p', 'h']

sns.lineplot(x=x[1:], y=Dual, color='orange', marker='p', label='Tda_Adapter')
sns.lineplot(x=x[0:1], y=MCM, color='purple', marker='v', label='Zero-shot MCM')
sns.lineplot(x=x[1:], y=CoOp, color='blue', marker='o', label='CoOp')
sns.lineplot(x=x[1:], y=LoCoOp, color='red', marker='h', label='LoCoOp')
# sns.lineplot(x=x[1:], y=tipkclx, color='orange', marker='p', label='Tip-Adapter+KCL*')
# sns.lineplot(x=x[1:], y=tipkcl, color='purple', marker='v', label='Tip-Adapter+KCL')

plt.annotate("MCM", xy=(0, 59.91), textcoords="offset points", xytext=(0, - 15), ha='center')

plt.grid(alpha=0.4)
plt.yticks([30, 35, 40, 45])
plt.xticks([0, 1, 2, 4, 8, 16])
plt.grid(alpha=0.4)
plt.xlabel('Shots')
plt.ylabel('AUROC')
plt.title('AUROC on Texture dataset')
plt.savefig('./texture_free_auroc.png')