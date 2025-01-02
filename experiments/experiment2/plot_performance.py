import pandas
import numpy
from matplotlib import pyplot as plt

def loadResults(type):

    if type == 'Traditional Features':
        LR = pandas.read_csv(f'results/LogReg_trad_results.csv')
        KNN = pandas.read_csv(f'results/KNN_trad_results.csv')
        SVM_Lin = pandas.read_csv(f'results/SVC_lin_trad_results.csv')
        SVM_cos = pandas.read_csv(f'results/SVC_cos_trad_results.csv')
        RF = pandas.read_csv(f'results/RF_trad_results.csv')
        Ada = pandas.read_csv(f'results/Ada_trad_results.csv')

    if type == 'Gabor Features':
        LR = pandas.read_csv(f'results/LogReg_gabor_results.csv')
        KNN = pandas.read_csv(f'results/KNN_gabor_results.csv')
        SVM_Lin = pandas.read_csv(f'results/SVC_lin_gabor_results.csv')
        SVM_cos = pandas.read_csv(f'results/SVC_cos_gabor_results.csv')
        RF = pandas.read_csv(f'results/RF_gabor_results.csv')
        Ada = pandas.read_csv(f'results/Ada_gabor_results.csv')

    if type == 'Deep Features (ImageNet)':
        LR = pandas.read_csv(f'results/LogReg_deep_ImageNet_results.csv')
        KNN = pandas.read_csv(f'results/KNN_deep_ImageNet_results.csv')
        SVM_Lin = pandas.read_csv(f'results/SVC_lin_deep_ImageNet_results.csv')
        SVM_cos = pandas.read_csv(f'results/SVC_cos_deep_ImageNet_results.csv')
        RF = pandas.read_csv(f'results/RF_deep_ImageNet_results.csv')
        Ada = pandas.read_csv(f'results/Ada_deep_ImageNet_results.csv')

    if type == 'Deep Features (OrganAMNIST)':
        LR = pandas.read_csv(f'results/LogReg_deep_MNIST_results.csv')
        KNN = pandas.read_csv(f'results/KNN_deep_MNIST_results.csv')
        SVM_Lin = pandas.read_csv(f'results/SVC_lin_deep_MNIST_results.csv')
        SVM_cos = pandas.read_csv(f'results/SVC_cos_deep_MNIST_results.csv')
        RF = pandas.read_csv(f'results/RF_deep_MNIST_results.csv')
        Ada = pandas.read_csv(f'results/Ada_deep_MNIST_results.csv')

    return LR, KNN, SVM_Lin, SVM_cos, RF, Ada

def plot(type, eval_type):
    if eval_type == 'val':
        key = 'auc_val_mean'
    elif eval_type == 'test':
        key = 'auc_test_mean'

    # Sample data
    x = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    LR, KNN, SVM_Lin, SVM_cos, RF, Ada = loadResults(type[0])
    y_LR0 = LR[key].to_list()[:-1]
    y_KNN0 = KNN[key].to_list()[:-1]
    y_SVM_Lin0 = SVM_Lin[key].to_list()[:-1]
    y_SVM_cos0 = SVM_cos[key].to_list()[:-1]
    y_RF0 = RF[key].to_list()[:-1]
    y_Ada0 = Ada[key].to_list()[:-1]

    LR, KNN, SVM_Lin, SVM_cos, RF, Ada = loadResults(type[1])
    y_LR1 = LR[key].to_list()[:-1]
    y_KNN1 = KNN[key].to_list()[:-1]
    y_SVM_Lin1 = SVM_Lin[key].to_list()[:-1]
    y_SVM_cos1 = SVM_cos[key].to_list()[:-1]
    y_RF1 = RF[key].to_list()[:-1]
    y_Ada1 = Ada[key].to_list()[:-1]

    LR, KNN, SVM_Lin, SVM_cos, RF, Ada = loadResults(type[2])
    y_LR2 = LR[key].to_list()[:-1]
    y_KNN2 = KNN[key].to_list()[:-1]
    y_SVM_Lin2 = SVM_Lin[key].to_list()[:-1]
    y_SVM_cos2 = SVM_cos[key].to_list()[:-1]
    y_RF2 = RF[key].to_list()[:-1]
    y_Ada2 = Ada[key].to_list()[:-1]

    LR, KNN, SVM_Lin, SVM_cos, RF, Ada = loadResults(type[3])
    y_LR3 = LR[key].to_list()[:-1]
    y_KNN3 = KNN[key].to_list()[:-1]
    y_SVM_Lin3 = SVM_Lin[key].to_list()[:-1]
    y_SVM_cos3 = SVM_cos[key].to_list()[:-1]
    y_RF3 = RF[key].to_list()[:-1]
    y_Ada3 = Ada[key].to_list()[:-1]

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, 2, figsize=(9, 9))

    # Plot in each subplot
    line1, = axs[0, 0].plot(x, y_LR0, label='Logistic Regression', color='tab:blue', linewidth=1.7)
    line1b, = axs[0, 0].plot(x, y_KNN0, label='K-Nearest Neighbors', color='tab:orange', linewidth=1.7)
    line1c, = axs[0, 0].plot(x, y_SVM_Lin0, label='SVM (Linear)', color='tab:red', linewidth=1.7)
    line1d, = axs[0, 0].plot(x, y_SVM_cos0, label='SVM (Cosine Similarity)', color='tab:green', linewidth=1.7)
    line1e, = axs[0, 0].plot(x, y_RF0, label='Random Forest', color='tab:brown', linewidth=1.7)
    line1f, = axs[0, 0].plot(x, y_Ada0, label='AdaBoost', color='tab:cyan', linewidth=1.7)
    axs[0, 0].axis((0, 100, 0.5, 1.01))
    axs[0, 0].set_title(type[0])
    #axs[0, 0].set_xticklabels([])
    axs[0, 0].grid(visible=True, alpha=0.4)

    line2, = axs[0, 1].plot(x, y_LR1, label='Logistic Regression', color='tab:blue', linewidth=1.7)
    line2b, = axs[0, 1].plot(x, y_KNN1, label='K-Nearest Neighbors', color='tab:orange', linewidth=1.7)
    line2c, = axs[0, 1].plot(x, y_SVM_Lin1, label='SVM (Linear)', color='tab:red', linewidth=1.7)
    line2d, = axs[0, 1].plot(x, y_SVM_cos1, label='SVM (Cosine Similarity)', color='tab:green', linewidth=1.7)
    line2e, = axs[0, 1].plot(x, y_RF1, label='Random Forest', color='tab:brown', linewidth=1.7)
    line2f, = axs[0, 1].plot(x, y_Ada1, label='AdaBoost', color='tab:cyan', linewidth=1.7)
    axs[0, 1].axis((0, 100, 0.5, 1.01))
    axs[0, 1].set_title(type[1])
    #axs[0, 1].set_xticklabels([])
    #axs[0, 1].set_yticklabels([])
    axs[0, 1].grid(visible=True, alpha=0.4)


    line3, = axs[1, 0].plot(x, y_LR2, label='Logistic Regression', color='tab:blue', linewidth=1.7)
    line3b, = axs[1, 0].plot(x, y_KNN2, label='K-Nearest Neighbors', color='tab:orange', linewidth=1.7)
    line3c, = axs[1, 0].plot(x, y_SVM_Lin2, label='SVM (Linear)', color='tab:red', linewidth=1.7)
    line3d, = axs[1, 0].plot(x, y_SVM_cos2, label='SVM (Cosine Similarity)', color='tab:green', linewidth=1.7)
    line3e, = axs[1, 0].plot(x, y_RF2, label='Random Forest', color='tab:brown', linewidth=1.7)
    line3f, = axs[1, 0].plot(x, y_Ada2, label='AdaBoost', color='tab:cyan', linewidth=1.7)
    axs[1, 0].axis((0, 100, 0.5, 1.01))
    axs[1, 0].set_title(type[2])
    #axs[1, 0].set_xticklabels([])
    axs[1, 0].grid(visible=True, alpha=0.4)

    line4, = axs[1, 1].plot(x, y_LR3, label='Logistic Regression', color='tab:blue', linewidth=1.7)
    line4b, = axs[1, 1].plot(x, y_KNN3, label='K-Nearest Neighbors', color='tab:orange', linewidth=1.7)
    line4c, = axs[1, 1].plot(x, y_SVM_Lin3, label='SVM (Linear)', color='tab:red', linewidth=1.7)
    line4d, = axs[1, 1].plot(x, y_SVM_cos3, label='SVM (Cosine Similarity)', color='tab:green', linewidth=1.7)
    line4e, = axs[1, 1].plot(x, y_RF3, label='Random Forest', color='tab:brown', linewidth=1.7)
    line4f, = axs[1, 1].plot(x, y_Ada3, label='AdaBoost', color='tab:cyan', linewidth=1.7)
    axs[1, 1].axis((0, 100, 0.5, 1.01))
    axs[1, 1].set_title(type[3])
    axs[1, 1].set_yticklabels([])
    axs[1, 1].grid(visible=True, alpha=0.4)

    # Add a common legend and labels for all subplots
    # We need to get the labels and lines from the subplots
    lines = [line1, line1b, line1c, line1d, line1e, line1f]
    labels = [line.get_label() for line in lines]
    leg = fig.legend(lines, labels, loc='upper center', ncol=3)
    fig.supxlabel('Number of features ($s$)')
    fig.supylabel('Average AUC')

    # Set the linewidth of each legend object
    for legobj in leg.legend_handles:
        legobj.set_linewidth(3.0)
    # Show the plot
    plt.show()

def main():
    type = ['Traditional Features', 'Gabor Features', 'Deep Features (ImageNet)', 'Deep Features (OrganAMNIST)']

    plot(type, eval_type='val')
    plot(type, eval_type='test')


main()