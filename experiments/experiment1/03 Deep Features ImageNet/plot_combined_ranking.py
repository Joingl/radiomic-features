import pandas
import seaborn as sns
import matplotlib.pylab as plt

pandas.set_option('display.max_rows', 500)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)

df = pandas.read_csv('CSVs/combined ranking.csv')

df['feature class'] = 'deep'
df = df.rename(columns={'ICC': 'Stability (ICC)', 'Average AUC': 'Discr. Power (AUC)'})

plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(figsize=(9, 9))
plt.title('Deep features')
ax.set_xlim(0, 1)
ax.set_ylim(0.5, 1)
sns.scatterplot(data=df, x='Stability (ICC)', y='Discr. Power (AUC)', style='feature class', hue='feature class', s=25, markers=['s'], palette=['tab:green'], legend=False)
ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize=20)
plt.plot([0.75, 0.75], [0, 1], color='tab:green', linestyle='dashed', linewidth=1.5, alpha=0.5)
plt.plot([0, 1], [0.8, 0.8], color='tab:green', linestyle='dashed', linewidth=1.5, alpha=0.5)
ax.axhspan(ymin=0.8, ymax=1, xmin=0.75, xmax=1, color='tab:green', alpha=0.3)

plt.grid()
plt.show()
