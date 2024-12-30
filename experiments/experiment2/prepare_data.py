import pandas
columns = ['group', 'scan_id', 'class']

def join_CTP4R_data():
    df1 = pandas.read_csv('../../feature extraction/ctp4r/01 Traditional Features/CSVs/traditional features CTP4R clean.csv')
    df2 = pandas.read_csv('../../feature extraction/ctp4r/02 Gabor Features/CSVs/gabor features CTP4R clean.csv')
    df3 = pandas.read_csv('../../feature extraction/ctp4r/03 Deep Features ImageNet/CSVs/deep features CTP4R ImageNet clean.csv')
    df4 = pandas.read_csv('../../feature extraction/ctp4r/04 Deep Features MNIST/CSVs/deep features CTP4R MNIST clean.csv')

    #Rename deep features:
    cols = df3.columns
    for c in cols[3:]:
        df3 = df3.rename(columns={f'{c}': f'{c}_ImageNet'})

    cols = df4.columns
    for c in cols[3:]:
        df4 = df4.rename(columns={f'{c}': f'{c}_MNIST'})

    #Join feature sets:
    df_new = df1.join(df2.iloc[:, 4:])
    df_new = df_new.join(df3.iloc[:, 3:])
    df_new = df_new.join(df4.iloc[:, 3:])

    # Drop mask size:
    df_new = df_new.drop(columns=['mask size'])

    #Drop all classes except normal and metastasis:
    df_new = df_new[df_new['class'].isin(['n1', 'n2', 'met'])]

    print(df_new.shape)
    print(df_new.columns)

    df_new.to_csv('data/all features CTP4R.csv', index=False)

    print(df_new.isnull().values.any())

def join_CRLM_data():
    df1 = pandas.read_csv('../../feature extraction/crlm/01 Traditional Features/CSVs/traditional features CRLM clean.csv')
    df2 = pandas.read_csv('../../feature extraction/crlm/02 Gabor Features/CSVs/gabor features CRLM clean.csv')
    df3 = pandas.read_csv('../../feature extraction/crlm/03 Deep Features ImageNet/CSVs/deep features CRLM ImageNet clean.csv')
    df4 = pandas.read_csv('../../feature extraction/crlm/04 Deep Features MNIST/CSVs/deep features CRLM MNIST clean.csv')

    #Rename deep features:
    cols = df3.columns
    for c in cols[3:]:
        df3 = df3.rename(columns={f'{c}': f'{c}_ImageNet'})

    cols = df4.columns
    for c in cols[3:]:
        df4 = df4.rename(columns={f'{c}': f'{c}_MNIST'})

    #Join feature sets:
    df_new = df1.join(df2.iloc[:, 4:])
    df_new = df_new.join(df3.iloc[:, 3:])
    df_new = df_new.join(df4.iloc[:, 3:])

    # Drop mask size:
    df_new = df_new.drop(columns=['mask size'])

    print(df_new.shape)
    print(df_new.columns)

    df_new.to_csv('data/all features CRLM.csv', index=False)

    print(df_new.isnull().values.any())

def get_feature_ranking():
    df = pandas.read_csv('../../feature extraction/crlm/01 Traditional Features/CSVs/AUC discr power.csv')
    df.to_csv('data/traditional feature ranking.csv')

    df = pandas.read_csv('../../feature extraction/crlm/02 Gabor Features/CSVs/AUC discr power.csv')
    df.to_csv('data/gabor feature ranking.csv')

    df = pandas.read_csv('../../feature extraction/crlm/03 Deep Features ImageNet/CSVs/AUC discr power.csv')
    df['feature'] = df['feature'] + '_ImageNet'
    df.to_csv('data/deep feature ranking ImageNet.csv')

    df = pandas.read_csv('../../feature extraction/crlm/04 Deep Features MNIST/CSVs/AUC discr power.csv')
    df['feature'] = df['feature'] + '_MNIST'
    df.to_csv('data/deep feature ranking MNIST.csv')

def main():
    join_CTP4R_data()
    join_CRLM_data()
    get_feature_ranking()

main()