import pydicom
import numpy
import json

scan_path = '01 scan/'
seg_path = '02 seg/'
classes = ['n1', 'n2', 'c1', 'c2', 'hem', 'met']

def generate_NumPy_images():
    scan_out_path = '01 scan numpy/'
    mask_out_path = '02 seg numpy/'

    f = open('largest_cross_sections.json')
    largest_cs = json.load(f)

    for k, v in largest_cs.items():
        dicom_filename_scan = f'{k[:3]}-1-{k[-3:]}.dcm'
        dicom_filename_mask = f'{v[:7]}.dcm'
        mask_idx = int(v[8:])
        print(dicom_filename_scan, dicom_filename_mask, mask_idx)
        scan = pydicom.dcmread(scan_path + dicom_filename_scan).pixel_array
        mask = pydicom.dcmread(seg_path + dicom_filename_mask).pixel_array[mask_idx]
        #print(scan.shape)
        #print(mask.shape)

        numpy.save(scan_out_path + k + '.npy', scan)
        numpy.save(mask_out_path + v + '.npy', mask)

def main():
    generate_NumPy_images()

main()