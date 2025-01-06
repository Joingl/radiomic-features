import os
import pandas
import numpy
import csv
import json
from PIL import Image
from helper.crop_image import cropImageTo96x96
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

f = open('../../data/ctp4r/largest_cross_sections.json')
largest_cs = json.load(f)

def write_header_to_csv(dict):
    with open("CSVs/deep features CTP4R ImageNet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, dict.keys())
        w.writeheader()

def add_row_to_csv(dict):
    with open('CSVs/deep features CTP4R ImageNet.csv', "a", newline="") as f:
        w = csv.DictWriter(f, dict.keys())
        w.writerow(dict)

def normalize_ImageNet(x):
    z = (x - x.min()) / (x.max() - x.min()) * 255
    return z

def load_ImageNet():
    model = models.resnet50(weights='IMAGENET1K_V1')
    return model

def extract_features_ImageNet(img, model):
    image_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3)])
    img = image_transforms(img)

    weights = ResNet50_Weights.IMAGENET1K_V1
    weights_transform = weights.transforms()
    img = weights_transform(img)

    img = img.unsqueeze(0)

    return_nodes = {'avgpool': 'average pooling layer'}
    extractor = create_feature_extractor(model, return_nodes=return_nodes)
    out = extractor(img)

    flatten = nn.Flatten()
    feature_vector = flatten(out['average pooling layer'])[0]

    res = {}
    for idx, i in enumerate(feature_vector):
        res[f'ft{idx}'] = round(i.item(), 6)

    return res

def iterate():
    scan_path = '../../data/ctp4r/01 scan numpy/'
    seg_path = '../../data/ctp4r/02 seg numpy/'
    first = True

    #Load model:
    imagenet_model = load_ImageNet()

    for k, v in largest_cs.items():
        print(k)

        #Load and crop images:
        scan = numpy.load(scan_path + k + '.npy')
        mask = numpy.load(seg_path + v + '.npy')

        scan = normalize_ImageNet(scan)
        scan = cropImageTo96x96(scan, mask)
        scan = scan.astype('float32')

        #Convert NumPy array to tiff file:
        img = Image.fromarray(scan)
        img.save('tmp.tiff')
        img = Image.open('tmp.tiff')


        result = extract_features_ImageNet(img, imagenet_model)
        result['scan_id'] = k
        result['group'] = k[0]
        result['class'] = k[4:-4]

        if first:
            write_header_to_csv(result)
            first = False

        add_row_to_csv(result)

        os.remove('tmp.tiff')


def formatCSV(df, filename):
    column_to_move = df.pop("class")
    df.insert(0, "class", column_to_move)
    column_to_move = df.pop("scan_id")
    df.insert(0, "scan_id", column_to_move)
    column_to_move = df.pop("group")
    df.insert(0, "group", column_to_move)

    df.to_csv(f'{filename} clean.csv', index=False)

def extract_features():
    iterate()

    df = pandas.read_csv('CSVs/deep features CTP4R ImageNet.csv')
    formatCSV(df, 'CSVs/deep features CTP4R ImageNet')