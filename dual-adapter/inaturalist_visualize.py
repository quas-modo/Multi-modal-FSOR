import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(dpi=500)
x = [0, 1, 2, 4, 8, 16]

CoOp = [91.26, 92.12, 92.60, 92.96, 94.43]
LoCoOp = [92.49, 92.76, 93.93, 94.60, 95.45]
Dual = [94.64, 94.64, 94.97, 94.63, 95.22]
MCM = [94.61]

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

plt.annotate("MCM", xy=(0, 94.61), textcoords="offset points", xytext=(0, -15), ha='center')

plt.grid(alpha=0.4)
plt.yticks([91, 92, 93, 94, 95, 96])
plt.xticks([0, 1, 2, 4, 8, 16])
plt.grid(alpha=0.4)
plt.xlabel('Shots')
plt.ylabel('AUROC')
plt.title('AUROC on iNaturalist dataset')
plt.savefig('./inaturalist_free_auroc.png')