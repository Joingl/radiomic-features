import os
import shutil

#Insert paths where the downloaded data is stored
path_ctp4r = 'D:/Radiomics Data/CT-Phantom4Radiomics/manifest-1681918831927/CT-Phantom4Radiomics/SPHNQA4IQI/'
path_crlm = 'D:/Radiomics Data/CRLM/manifest-1669817128730/Colorectal-Liver-Metastases/'

def move_scans_and_rois_ctp4r():
    path_out_scan_ctp4r = 'ctp4r/01 scan/'  # Path where scans will be stored
    path_out_seg_ctp4r = 'ctp4r/02 seg/'  # Path where segmentation will be stored

    dirs = [dir for dir in os.listdir(path_ctp4r)]
    for dir in dirs:
        dirs2 = [dir2 for dir2 in os.listdir(path_ctp4r + dir)]
        scan_folder = dirs2[1]
        seg_folder = dirs2[0]
        series = scan_folder[:3]
        files = [file for file in os.listdir(path_ctp4r + dir + '/' + scan_folder)]
        for f in files:
            p = f'{path_ctp4r}/{dir}/{scan_folder}/{f}'
            name = f'{series}-{f}'
            print(name)
            shutil.copyfile(p, path_out_scan_ctp4r + name)

        files2 = [file2 for file2 in os.listdir(path_ctp4r + dir + '/' + seg_folder)]
        for f2 in files2:
            p2 = f'{path_ctp4r}/{dir}/{seg_folder}/{f2}'
            name = f'{series}-{f2}'
            print(name)
            shutil.copyfile(p2, path_out_seg_ctp4r + name)

def move_scans_and_rois_crlm():
    path_out_scan_crlm = 'crlm/01 scan/'
    path_out_seg_crlm = 'crlm/02 seg/'

    dirs = [dir for dir in os.listdir(path_crlm)]
    for dir in dirs:
        dirs2 = [dir2 for dir2 in os.listdir(path_crlm + dir)]
        for dir2 in dirs2:
            dirs3 = [dir3 for dir3 in os.listdir(path_crlm + dir + '/' + dir2)]
            print(dirs3)
            if 'Seg' in dirs3[0]:
                scan_folder = dirs3[1]
                seg_folder = dirs3[0]

            elif 'Seg' in dirs3[1]:
                scan_folder = dirs3[0]
                seg_folder = dirs3[1]

            series = dir[-4:]
            files = [file for file in os.listdir(path_crlm + dir + '/' + dir2 + '/' + scan_folder)]
            for f in files:
                p = f'{path_crlm}{dir}/{dir2}/{scan_folder}/{f}'
                print(p)
                name = f'{series}-{f}'
                print(name)
                shutil.copyfile(p, path_out_scan_crlm + name)

            files2 = [file2 for file2 in os.listdir(path_crlm + dir + '/' + dir2 + '/' + seg_folder)]
            for f2 in files2:
                p2 = f'{path_crlm}{dir}/{dir2}/{seg_folder}/{f2}'
                print(p2)
                name = f'{series}-{f2}'
                print(name)
                shutil.copyfile(p2, path_out_seg_crlm + name)

def main():
    move_scans_and_rois_ctp4r()
    move_scans_and_rois_crlm()

main()