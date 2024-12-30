import numpy
import pandas
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

numpy.set_printoptions(suppress=True)
pandas.set_option('future.no_silent_downcasting', True)
pandas.set_option('display.max_rows', 1400)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)
pandas.set_option('display.float_format', lambda x: '%.4f' % x)

def cosine_kernel(X, Y):
    return cosine_similarity(X, Y)

def loadFeaturesRankedByDiscrPower(type, size):
    if type == 'trad':
        df = pandas.read_csv('data/traditional feature ranking.csv')
    if type == 'gabor':
        df = pandas.read_csv('data/gabor feature ranking.csv')
    if type == 'deep_ImageNet':
        df = pandas.read_csv('data/deep feature ranking ImageNet.csv')
    if type == 'deep_MNIST':
        df = pandas.read_csv('data/deep feature ranking MNIST.csv')

    all_features = df['feature'].to_list()

    if size == 'max':
        return all_features
    else:
        features = all_features[:size]
        return features

def loadCRLM(features):
    columns = ['series', 'scan_id', 'class']

    data = pandas.read_csv('data/all features CRLM.csv')

    data['class'] = data['class'].str[0]
    data['class'] = data['class'].replace('r', 0)
    data['class'] = data['class'].replace('t', 1)
    data = data[columns + features]

    return data

def loadCTP4R(features):
    columns = ['group', 'scan_id', 'class']

    data = pandas.read_csv('data/all features CTP4R.csv')

    data['class'] = data['class'].replace('n1', 0)
    data['class'] = data['class'].replace('n2', 0)
    data['class'] = data['class'].replace('met', 1)
    data = data[columns + features]

    return data

def getMetrics(features, classifier, random_states):
    data_train = loadCRLM(features)
    data_test = loadCTP4R(features)

    X = data_train.iloc[:, 3:].to_numpy()
    y = data_train['class'].to_list()

    accs_val = []
    aucs_val = []
    accs_test = []
    aucs_test = []
    for r in random_states:
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.75, shuffle=True, random_state=r)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        match classifier:
            case 'LR':
                classifier = LogisticRegression(max_iter=300, C=0.5, penalty='l2', solver='lbfgs', random_state=42)
                model_name = 'LogReg'
            case 'KNN':
                classifier = KNeighborsClassifier(n_neighbors=100)
                model_name = 'KNN'
            case 'SVC_lin':
                classifier = SVC(kernel='linear', C=1, probability=True, random_state=42)
                model_name = 'SVC_lin'
            case 'SVC_cos':
                classifier = SVC(kernel=cosine_kernel, C=1, probability=True, random_state=42)
                model_name = 'SVC_cos'
            case 'RF':
                classifier = RandomForestClassifier(n_estimators=100, max_depth=3, criterion='gini', random_state=42)
                model_name = 'RF'
            case 'Ada':
                classifier = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, algorithm='SAMME', random_state=42) # [*]
                model_name = 'Ada'


        classifier.fit(X_train, y_train)

        #Evaluate on Validation set:
        y_pred_val = classifier.predict(X_val)
        y_probabilities_val = classifier.predict_proba(X_val)[:, 1]

        acc_val = balanced_accuracy_score(y_val, y_pred_val)
        accs_val.append(acc_val)

        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_probabilities_val, pos_label=1)
        AUC_val = metrics.auc(fpr, tpr)
        aucs_val.append(AUC_val)

        #Evaluate on Test set:
        X_test = data_test.iloc[:, 3:].to_numpy()
        y_test = numpy.array(data_test['class'].to_list())
        X_test = scaler.transform(X_test)

        y_pred = classifier.predict(X_test)
        y_probabilities = classifier.predict_proba(X_test)[:, 1]

        acc = balanced_accuracy_score(y_test, y_pred)
        accs_test.append(acc)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_probabilities, pos_label=1)
        AUC = metrics.auc(fpr, tpr)
        aucs_test.append(AUC)

    return accs_val, aucs_val, accs_test, aucs_test, model_name

def getResults(type, classifier, random_states):
    sizes = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 'max']

    res = {}

    for s in sizes:
        features = loadFeaturesRankedByDiscrPower(type, s)

        metrics = getMetrics(features, classifier, random_states)
        model_name = metrics[4]
        list = [f'{type}', s, numpy.array(metrics[1]).mean(), numpy.array(metrics[1]).std(), numpy.array(metrics[3]).mean(), numpy.array(metrics[3]).std(), numpy.array(metrics[1]), numpy.array(metrics[3])]
        res[f'{type}_{s}'] = list

    df = pandas.DataFrame().from_dict(res, orient='index', columns=['type', 'size', 'auc_val_mean', 'auc_val_std', 'auc_test_mean', 'auc_test_std', 'auc_val', 'auc_test'])
    print(df[['type', 'size', 'auc_val_mean', 'auc_val_std', 'auc_test_mean', 'auc_test_std']])

    df.to_csv(f'results/{model_name}_{type}_results.csv', index=False)


def run(classifier):
    random_states = numpy.load('random_seeds.npy')
    getResults('trad', classifier, random_states)
    getResults('gabor', classifier, random_states)
    getResults('deep_ImageNet', classifier, random_states)
    getResults('deep_MNIST', classifier, random_states)

def main():
    classifiers = ['LR', 'KNN', 'SVC_lin', 'SVC_cos', 'RF', 'Ada']
    for c in classifiers:
        run(c)

main()