import extract_traditional_features as trad
#import extract_gabor_features as gabor
import extract_deep_features_ImageNet as deepImageNet
import extract_deep_features_MNIST as deepMNIST
import compute_ranking as ranking

def main():
    trad.extract_features()
    #gabor.extract_features()
    deepImageNet.extract_features()
    deepMNIST.extract_features()
    ranking.compute_AUC_ranking()

main()