import pingouin as pg
import pandas
from sklearn import metrics
import numpy

groups = [1, 2, 3, 4, 5, 6, 7, 8]
tissue_classes = ['n', 'c', 'h', 'm']
tissue_classes_idx = [0, 1, 2, 3]
tissue_names = ['norm', 'cyst', 'hema', 'meta']

def compute_ICC(df):
    cols = df.columns.values
    d = {'feature': cols[3:]}
    results = pandas.DataFrame(d)
    results['feature'] = cols[3:]

    ICCs = []
    for c in cols: #Compute ICC for each feature
        if c not in ['group', 'scan_id', 'class']:
            res = pg.intraclass_corr(data=df, targets='class', raters='group', ratings=c)
            ICC = res[res['Type'] == 'ICC2'].iloc[0]['ICC'].round(4)
            ICCs.append(ICC)

    results['ICC'] = ICCs

    results.to_csv('CSVs/ICC2 stability.csv', index=False)

def compute_AUC_per_group(data, A, B):
    data.iloc[:, 2] = data.iloc[:, 2].str[0]  # renames classes to single characters, which groups n1 & n2, c1 & c2

    df = data[(data['class'] == A) | (data['class'] == B)]
    cols = df.columns.values
    res_list = []

    for c in cols:
        if c not in ['group', 'scan_id', 'class']:
            AUC_list = []
            AUC_list2 = []
            for g in groups:
                df_tmp = df[df['group'] == g]

                y_true = df_tmp['class'].to_numpy()
                y_score = df_tmp[c]

                fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=A)
                AUC = metrics.auc(fpr, tpr)

                fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=B)
                AUC2 = metrics.auc(fpr, tpr)

                AUC_list.append(AUC)
                AUC_list2.append(AUC2)

            res = numpy.round(numpy.array(AUC_list).mean(), 4)
            res2 = numpy.round(numpy.array(AUC_list2).mean(), 4)

            if res > res2:
                res_list.append(res)
            else:
                res_list.append(res2)

    return res_list

def compute_AUC(df):
    cols = df.columns.values
    d = {'feature': cols[3:]}
    results = pandas.DataFrame(d)

    for c1 in tissue_classes_idx:
        for c2 in tissue_classes_idx:
            if c1 < c2:
                print(f'{c1} vs. {c2}')
                partial_res_list = compute_AUC_per_group(df, tissue_classes[c1], tissue_classes[c2])
                results[f'{tissue_names[c1]} vs. {tissue_names[c2]}'] = partial_res_list

    results.to_csv('CSVs/AUC discr power.csv', index=False)

def compute_combined_score(df_ICC, df_AUC):
    df_ICC = df_ICC.set_index('feature')
    df_AUC = df_AUC.set_index('feature')

    #Get Means
    df_AUC['mean'] = df_AUC.mean(axis=1).round(4)

    #Combine Means
    df_means = pandas.DataFrame()
    df_means['ICC'] = df_ICC['ICC']
    df_means['Average AUC'] = df_AUC['mean']
    df_means['Combined Score'] = ((df_means['ICC'] + df_means['Average AUC']) / 2).round(4)

    #Ranking
    df_ranking = df_means.sort_values(by='Combined Score', ascending=False)
    df_ranking.to_csv('CSVs/combined ranking.csv', index=True)

def main():
    df = pandas.read_csv('../../../feature extraction/ctp4r/CSVs/deep features CTP4R MNIST clean.csv')
    compute_ICC(df)
    compute_AUC(df)

    df_ICC = pandas.read_csv('CSVs/ICC2 stability.csv')
    df_AUC = pandas.read_csv('CSVs/AUC discr power.csv')

    compute_combined_score(df_ICC, df_AUC)

main()