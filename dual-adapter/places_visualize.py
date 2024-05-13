import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(dpi=500)
x = [0, 1, 2, 4, 8, 16]

CoOp = [89.09, 88.98, 89.15, 89.76, 89.74]
LoCoOp = [91.07, 90.38, 90.32, 90.53, 90.64]
Dual = [90.50, 90.74, 90.74, 90.59, 91.55]
MCM = [89.76]

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

plt.annotate("MCM", xy=(0, 89.76), textcoords="offset points", xytext=(0, -15), ha='center')

plt.grid(alpha=0.4)
plt.yticks([89, 90, 91, 92])
plt.xticks([0, 1, 2, 4, 8, 16])
plt.grid(alpha=0.4)
plt.xlabel('Shots')
plt.ylabel('AUROC')
plt.title('AUROC on Places dataset')
plt.savefig('./places_free_auroc.png')