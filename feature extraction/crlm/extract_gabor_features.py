import csv
import pandas
from radiomics import featureextractor
import SimpleITK as sitk
import json
import numpy
import cv2
from helper.apply_gabor_filter import apply_filter

scan_path = '../../data/crlm/01 scan numpy/'
seg_path = '../../data/crlm/02 seg numpy/'

f = open('../../data/crlm/largest_cross_sections.json')
largest_cs = json.load(f)

pandas.set_option('display.max_rows', 1400)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)

def has_more_than_one_dimension(mask):
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

def write_header_to_csv(dict):
    with open("CSVs/gabor features CRLM.csv", "w", newline="") as f:
        w = csv.DictWriter(f, dict.keys())
        w.writeheader()

def add_row_to_csv(dict):
    with open("CSVs/gabor features CRLM.csv", "a", newline="") as f:
        w = csv.DictWriter(f, dict.keys())
        w.writerow(dict)

def erode_mask(mask):
    kernel = numpy.ones((3, 3), numpy.uint8)
    mask_eroded = cv2.erode(mask, kernel=kernel, iterations=4)
    if numpy.nonzero(mask_eroded)[0].shape[0] > 100 and has_more_than_one_dimension(mask_eroded):
        return mask_eroded
    else:
        return mask

def extract_features():
    extractor = featureextractor.RadiomicsFeatureExtractor('settings_gabor.yaml')

    first = True
    for k, v in largest_cs.items():
        print(k)

        #Load and crop image and mask for more efficiency:
        scan = numpy.load(scan_path + k + '.npy')
        mask = numpy.load(seg_path + v + '.npy')

        #Apply Gabor filter:
        filtered_images = apply_filter(scan)

        #Erode mask:
        mask = erode_mask(mask)

        #Get mask size:
        seg_size = len(numpy.nonzero(mask)[0])

        #Convert to SITK image:
        mask = sitk.GetImageFromArray(mask)

        for idx, i in enumerate(filtered_images): #Iterate over all 24 filtered images
            img = sitk.GetImageFromArray(i)

            result = extractor.execute(img, mask)
            result['scan_id'] = k
            result['series'] = k[:4]
            result['class'] = k[5:8]
            result['mask size'] = seg_size
            result['feature_index'] = idx

            if first:
                write_header_to_csv(result)
                first = False

            add_row_to_csv(result)

    df = pandas.read_csv('CSVs/gabor features CRLM.csv')

    format_CSV(df)

def format_CSV(df):
    templ = pandas.read_csv('CSVs/traditional features CRLM clean.csv')
    df2 = templ.iloc[:, :4]

    column_to_move = df.pop("mask size")
    df.insert(0, "mask size", column_to_move)
    column_to_move = df.pop("class")
    df.insert(0, "class", column_to_move)
    column_to_move = df.pop("scan_id")
    df.insert(0, "scan_id", column_to_move)
    column_to_move = df.pop("series")
    df.insert(0, "series", column_to_move)
    column_to_move = df.pop("feature_index")
    df.insert(0, "feature_index", column_to_move)

    cols = [col for col in df.columns if (col[0:9] == 'original_') | (col in ['series', 'scan_id', 'class', 'mask size', 'feature_index'])]
    df = df[cols]
    cols2 = cols[5:]

    for c in cols2:
        for i in range(24): #i = 0,...,23
            df_idx = df[df['feature_index'] == i]
            df_idx = df_idx.set_index(templ.index)
            df2[f'{c}_idx{i}'] = df_idx[c]

    for i in range(24):
        df2[f'original_firstorder_Energy_idx{i}'] = df2[f'original_firstorder_Energy_idx{i}'] / df2['mask size']

    df2.to_csv('CSVs/gabor features CRLM clean.csv', index=False)