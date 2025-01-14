import pandas
import seaborn as sns
import matplotlib.pylab as plt

df = pandas.read_csv('CSVs/combined ranking.csv')
df['feature type'] = df['feature'].str[9:12]
df['feature type'] = df['feature type'].str.replace('fir', 'First order')
df['feature type'] = df['feature type'].str.replace('glc', 'Texture')
df['feature type'] = df['feature type'].str.replace('glr', 'Texture')
df['feature type'] = df['feature type'].str.replace('gls', 'Texture')
df['feature type'] = df['feature type'].str.replace('ngt', 'Texture')
df['feature type'] = df['feature type'].str.replace('gld', 'Texture')

df = df.rename(columns={'ICC': 'Stability (ICC)', 'Average AUC': 'Discr. Power (AUC)'})

plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(figsize=(9, 9))
plt.title('Traditional features')
ax.set_xlim(0, 1)
ax.set_ylim(0.5, 1)

sns.scatterplot(data=df[df['feature type'] == 'First order'], x='Stability (ICC)', y='Discr. Power (AUC)', style='feature type', s=150, legend='brief')
sns.scatterplot(data=df[df['feature type'] == 'Texture'], x='Stability (ICC)', y='Discr. Power (AUC)', style='feature type', s=150, hue='feature type', markers=['v'], palette=['tab:blue'], legend='brief')
ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize=20)
ax.legend(title='')
plt.plot([0.75, 0.75], [0, 1], color='tab:blue', linestyle='dashed', linewidth=1.5, alpha=0.5)
plt.plot([0, 1], [0.8, 0.8], color='tab:blue', linestyle='dashed', linewidth=1.5, alpha=0.5)
ax.axhspan(ymin=0.8, ymax=1, xmin=0.75, xmax=1, color='tab:blue', alpha=0.3)

plt.grid()
plt.show()
