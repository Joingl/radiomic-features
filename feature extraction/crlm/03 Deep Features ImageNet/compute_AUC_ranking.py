import pandas
from sklearn import metrics
import numpy

tissue_classes = ['r', 't']
tissue_classes_idx = [0, 1]
tissue_names = ['rem', 'tum']

def compute_AUC_for_all_features(data, A, B):
    data.iloc[:, 2] = data.iloc[:, 2].str[0]  # renames classes to single characters

    df = data[(data['class'] == A) | (data['class'] == B)]
    cols = df.columns.values
    res_list = []

    for c in cols:
        if c not in ['series', 'scan_id', 'class', 'mask size']:
            AUC_list = []
            AUC_list2 = []

            df_tmp = df

            y_true = df_tmp['class'].to_numpy()
            y_score = df_tmp[c]

            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=A)
            AUC = metrics.auc(fpr, tpr)

            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=B)
            AUC2 = metrics.auc(fpr, tpr)

            AUC_list.append(AUC)
            AUC_list2.append(AUC2)

            res = numpy.array(AUC_list).mean()
            res2 = numpy.array(AUC_list2).mean()
            #print(f'{c}: {res}')
            #print(f'{c}: {res2}')

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
                partial_res_list = compute_AUC_for_all_features(df, tissue_classes[c1], tissue_classes[c2])
                results[f'{tissue_names[c1]} vs. {tissue_names[c2]}'] = partial_res_list

    results.to_csv('CSVs/AUC discr power.csv', index=False)

def compute_AUC_ranking():
    df = pandas.read_csv('CSVs/AUC discr power.csv')
    df = df.sort_values(by='rem vs. tum', ascending=False)
    df.to_csv('CSVs/AUC discr power.csv', index=False)


def main():
    df = pandas.read_csv('CSVs/deep features CRLM ImageNet clean.csv')
    compute_AUC(df)
    compute_AUC_ranking()

main()