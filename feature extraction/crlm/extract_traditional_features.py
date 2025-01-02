import csv
import json
import numpy
import pandas
from radiomics import featureextractor
import SimpleITK as sitk

scan_path = '../../data/crlm/01 scan numpy/'
seg_path = '../../data/crlm/02 seg numpy/'

f = open('../../data/crlm/largest_cross_sections.json')
largest_cs = json.load(f)

def write_header_to_CSV(dict):
    with open("CSVs/traditional features CRLM.csv", "w", newline="") as f:
        w = csv.DictWriter(f, dict.keys())
        w.writeheader()

def add_row_to_CSV(dict):
    with open("CSVs/traditional features CRLM.csv", "a", newline="") as f:
        w = csv.DictWriter(f, dict.keys())
        w.writerow(dict)

def has_more_than_one_dimension(mask): # Detects if a mask only has one dimension. Can't be used for feature extraction then.
    dim_y_greater_one = False
    dim_x_greater_one = False

    first = mask.nonzero()[0][0]
    for i in mask.nonzero()[0][1:]:
        if i != first:
            dim_y_greater_one = True
            break

    first = mask.nonzero()[1][0]
    for i in mask.nonzero()[1][1:]:
        if i != first:
            dim_x_greater_one = True
            break

    return dim_y_greater_one and dim_x_greater_one

def format_CSV(df):
    column_to_move = df.pop("mask size")
    df.insert(0, "mask size", column_to_move)
    column_to_move = df.pop("class")
    df.insert(0, "class", column_to_move)
    column_to_move = df.pop("scan_id")
    df.insert(0, "scan_id", column_to_move)
    column_to_move = df.pop("series")
    df.insert(0, "series", column_to_move)


    cols = [col for col in df.columns if (col[0:9] == 'original_') | (col in ['series', 'scan_id', 'class', 'mask size'])]
    df = df[cols]

    df['original_firstorder_Energy'] = df['original_firstorder_Energy'] / df['mask size'] #Adjust energy feature to the size of the ROI

    df.to_csv('CSVs/traditional features CRLM clean.csv', index=False)

def extract_features():
    extractor = featureextractor.RadiomicsFeatureExtractor('settings.yaml')

    first = True
    for k, v in largest_cs.items():
        print(k)

        img = numpy.load(scan_path + k + '.npy')
        mask = numpy.load(seg_path + v + '.npy')
        seg_size = len(numpy.nonzero(mask)[0])

        img = sitk.GetImageFromArray(img)
        mask = sitk.GetImageFromArray(mask)

        result = extractor.execute(img, mask)
        result['scan_id'] = k
        result['series'] = k[:4]
        result['class'] = k[5:8]
        result['mask size'] = seg_size

        if first:
            write_header_to_CSV(result)
            first = False

        add_row_to_CSV(result)

    df = pandas.read_csv('CSVs/traditional features CRLM.csv')
    format_CSV(df)