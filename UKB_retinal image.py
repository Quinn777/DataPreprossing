import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import Counter
import os
import glob
import re
import shutil


def get_csv():
    dir1 = r"C:\Users\xiangk\Desktop\5-12 retinal image population with numeric indications.txt"
    dir2 = r"C:\Users\xiangk\Desktop\5-12 retinal image population with diagnosis.txt"
    # 打开标签文件

    # df1 = pd.read_csv(dir1, delimiter="\t")
    # df1.to_csv(r"C:\Users\xiangk\Desktop\5-12 retinal image population with numeric indications.csv", encoding='utf-8', index=False)
    #
    # df2 = pd.read_csv(dir2, delimiter="\t")
    # df2.to_csv(r"C:\Users\xiangk\Desktop\5-12 retinal image population with diagnosis.csv", encoding='utf-8', index=False)


    with open(dir2, 'r') as file:
        # 逐行读取文件内容
        lines = file.readlines()

    # 处理每行数据
    data1 = []
    i = 0
    for line in lines:
        # 移除换行符和空格
        line = line.strip()
        line = line.split()

        # 在这里可以对每个标签进行进一步处理或使用
        # 例如，打印标签
        a = np.array(line[2:])
        if i > 0 and (a == 'NA').all():
            data1.append(line[:2])
        i += 1

    with open(dir1, 'r') as file:
        # 逐行读取文件内容
        lines = file.readlines()

    data2 = []
    for line in tqdm(lines):
        # 移除换行符和空格
        line = line.strip()
        line = line.split()

        j = 0
        for data in data1:
            if line[0] == data[0]:
                if len(data1[j]) == 2:
                    data1[j].append(line[3])
            j += 1

    name = ['f.eid', 'fieldid', 'age']
    label = pd.DataFrame(columns=name, data=data1)
    label.to_csv(r"C:\Users\xiangk\Desktop\5-12 retinal image labels.csv",encoding='utf-8')

def get_sample_distribution():
    df = pd.read_csv(r"C:\Users\xiangk\Desktop\5-12 retinal image labels.csv")
    ages = df['age'].values
    print(pd.value_counts(ages))

def del_file():
    path = r'E:\ukb'
    for infile in tqdm(glob.glob(os.path.join(path, '*.tmp_bulk'))):
        os.remove(infile)


def motify_csv():
    df = pd.read_csv(r"C:\Users\xiangk\Desktop\5-12 retinal image labels.csv")
    ages = df['age'].values

    path = r'E:\ukb'
    new_path = r"F:\003_Datasets\uukb_normal_retinal_image"
    # for infile in tqdm(glob.glob(os.path.join(path, '*.png'))):
    #     os.remove(infile)
    #     file_name = infile.split("\\")[-1]
    #     f_eid = file_name.split("_")[0]
    #     field_id = re.sub(f_eid+"_", "", file_name, ).split(".")[0]
    #     for index, row in df.iterrows():
    #         # print(index)  # 输出每行的索引值
    #         if field_id == row['fieldid'] and f_eid == row['f.eid']:
    #             data.append([row['f.eid'], row['fieldid'], row['age'], file_name])
    i = 0
    data = []
    for index, row in tqdm(df.iterrows()):
        file_name = str(row['f.eid']) + "_" + row['fieldid'] + ".png"
        file_dir = os.path.join(path, file_name)
        if os.path.exists(file_dir):
            new_file_dir = os.path.join(new_path, file_name)
            shutil.copy(file_dir, new_file_dir)

            data.append([row['f.eid'], row['fieldid'], row['age'], file_name, int((int(row['age'])-40)/5)])



    # print(pd.value_counts(ages))
    name = ['f.eid', 'fieldid', 'age', 'path', 'label']
    label = pd.DataFrame(columns=name, data=data)
    label.to_csv(r"C:\Users\xiangk\Desktop\normal retinal image labels.csv", encoding='utf-8')

def read_fds():
    from oct_converter.readers import FDS
    import json
    # An example .fds file can be downloaded from the Biobank website:
    # https://biobank.ndph.ox.ac.uk/showcase/refer.cgi?id=30
    filepath = r'C:\Users\xiangk\Desktop\OCT\OCT/1000515_21014_0_0.fds'
    fds = FDS(filepath)

    oct_volume = fds.read_oct_volume()  # returns an OCT volume with additional metadata if available
    oct_volume.peek(show_contours=True)  # plots a montage of the volume, with layer segmentations is available
    oct_volume.save('fds_testing.avi')  # save volume as a movie
    oct_volume.save('fds_testing.png')  # save volume as a set of sequential images, fds_testing_[1...N].png
    oct_volume.save_projection('projection.png')  # save 2D projection

    fundus_image = fds.read_fundus_image()  # returns a  Fundus image with additional metadata if available
    fundus_image.save('fds_testing_fundus.jpg')

    metadata = fds.read_all_metadata(verbose=True)  # extracts all other metadata
    with open("fds_metadata.json", "w") as outfile:
        outfile.write(json.dumps(metadata, indent=4))
if __name__ == '__main__':
    # get_sample_distribution()
    # get_csv()
    # motify_csv()
    # read_fds()
    import cv2
    img = cv2.imread(r"C:\Users\xiangk\Desktop\1036443_21016_0_0.png")
    resized_img = cv2.resize(img, (224, 224))
    cv2.imshow('a',resized_img)
    cv2.waitKey(0)
    cv2.imwrite(r"C:\Users\xiangk\Desktop\1036443_21016_0_0_2.png", resized_img)

    # plt.imshow(image)
    # plt.savefig('test.png', format='png', dpi=200)  # 保存缩放后图片
