import pydicom
import numpy
import os
import json
import SimpleITK as sitk
from helper.harmonize_pixel_spacing import resample

scan_path = '01 scan/'
seg_path = '02 seg/'
classes = ['liv', 'rem', 'hep', 'por', 't01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11',
           't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25',
           't26', 't27', 't28', 't29', 't30', 't31', 't32', 't33', 't34', 't35']

def generate_numpy_crlm():
    scan_out_path = '01 scan numpy/'
    mask_out_path = '02 seg numpy/'

    f = open('largest_cross_sections.json')
    largest_cs = json.load(f)

    for k, v in largest_cs.items():
        dicom_filename_scan = f'{k[:4]}-1-{k[9:]}.dcm'
        dicom_filename_mask = f'{v[:8]}.dcm'
        mask_idx = int(v[9:])
        print(dicom_filename_scan, dicom_filename_mask, mask_idx)
        scan = pydicom.dcmread(scan_path + dicom_filename_scan).pixel_array
        mask = pydicom.dcmread(seg_path + dicom_filename_mask).pixel_array[mask_idx]
        #print(scan.shape)
        #print(mask.shape)

        numpy.save(scan_out_path + k + '.npy', scan)
        numpy.save(mask_out_path + v + '.npy', mask)