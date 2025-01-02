import pandas
import numpy
import csv
import json
import os
from PIL import Image
import PIL
from helper.crop_image import cropImageTo96x96
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

f = open('../../data/crlm/largest_cross_sections.json')
largest_cs = json.load(f)

def write_header_to_csv(dict):
    with open("CSVs/deep features CRLM MNIST.csv", "w", newline="") as f:
        w = csv.DictWriter(f, dict.keys())
        w.writeheader()

def add_row_to_csv(dict):
    with open('CSVs/deep features CRLM MNIST.csv', "a", newline="") as f:
        w = csv.DictWriter(f, dict.keys())
        w.writerow(dict)

def normalizeMNIST(x): #Standard and correct way IMO.
    z = (x - x.min()) / (x.max() - x.min()) * 255
    return z

def load_MNIST():
    weights = 'resnet50_224_1.pt'
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 11)
    model.load_state_dict(torch.load(weights, weights_only=True, map_location=torch.device('cpu')))
    return model

def extract_features_MNIST(img, model):

    data_transform = transforms.Compose( #Transforms as shown in code by MedMNIST: https://github.com/MedMNIST/experiments/blob/main/MedMNIST2D/train_and_eval_pytorch.py
        [transforms.Grayscale(num_output_channels=3),
         transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
         transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    img = data_transform(img)
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
    scan_path = '../../data/crlm/01 scan numpy/'
    seg_path = '../../data/crlm/02 seg numpy/'
    first = True

    #Load model:
    mnist_model = load_MNIST()

    for k, v in largest_cs.items():
        print(k)

        #Load and crop images:
        scan = numpy.load(scan_path + k + '.npy')
        mask = numpy.load(seg_path + v + '.npy')

        scan = normalizeMNIST(scan)
        scan = cropImageTo96x96(scan, mask)
        scan = scan.astype('float32')

        #Convert NumPy array to tiff file:
        img = Image.fromarray(scan)
        img.save('tmp.tiff')
        img = Image.open('tmp.tiff')

        result = extract_features_MNIST(img, mnist_model)
        result['scan_id'] = k
        result['series'] = k[:4]
        result['class'] = k[5:8]

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
    column_to_move = df.pop("series")
    df.insert(0, "series", column_to_move)

    df.to_csv(f'{filename} clean.csv', index=False)

def extract_features():
    iterate()

    df = pandas.read_csv('CSVs/deep features CRLM MNIST.csv')
    formatCSV(df, 'CSVs/deep features CRLM MNIST')
