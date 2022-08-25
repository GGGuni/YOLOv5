import os
import cv2
import math

tasks = ["VisDrone2019-DET-train", "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev"]
tt = ['train/', 'val/', 'test/']

"""
for changing label only
"""
# target_root = "./data/reifne_label_drone_pic"
# if not os.path.exists(target_root):
#     os.mkdir(target_root)
#     os.mkdir(target_root+'/images')
#     os.mkdir(target_root+'/images/train')
#     os.mkdir(target_root+'/images/val')
#     os.mkdir(target_root+'/images/test')
#     os.mkdir(target_root+'/labels')
#     os.mkdir(target_root+'/labels/train')
#     os.mkdir(target_root+'/labels/val')
#     os.mkdir(target_root+'/labels/test')

"""
for changing label and filtering size
"""
# target_root = "./data/reifne_label_size_drone_pic"
target_root = "./data/test/"

if not os.path.exists(target_root):
    os.mkdir(target_root)
    os.mkdir(target_root+'images')
    os.mkdir(target_root+'images/train')
    os.mkdir(target_root+'images/val')
    os.mkdir(target_root+'images/test')
    os.mkdir(target_root+'labels')
    os.mkdir(target_root+'labels/train')
    os.mkdir(target_root+'labels/val')
    os.mkdir(target_root+'labels/test')


for idx in range(len(tasks)):
    task = tasks[idx]
    ttt = tt[idx]
    root = "./data/original_dataset/"+task

    #changelabelonly target_l = "./data/reifne_label_drone_pic/labels/"+ttt
    target_ls = target_root+"labels/"+ttt

    #saveimg target_i = "./data/reifne_label_drone_pic/images/"+ttt
    #saveimg target_is = "./data/reifne_label_size_drone_pic/images/"+ttt

    img_path = root + "/images/"
    annot_path = root + "/annotations/"
    #changelabelonly annot_lpath = target_l
    annot_lspath = target_ls

    annot_list = os.listdir(annot_path)
    annot_list.sort()
    img_list = os.listdir(img_path)
    img_list.sort()

    for (n_a, n_i) in zip(annot_list, img_list):
        f_a = open(annot_path+n_a, 'r')
        f_a_lines = f_a.readlines()

        #changelabelonly f_a_l = open(annot_lpath+n_a, 'w')
        f_a_ls = open(annot_lspath+n_a, 'w')

        f_i = cv2.imread(img_path+n_i)
        #saveimg cv2.imwrite(target_i+'/'+n_i, f_i)
        #saveimg cv2.imwrite(target_is+'/'+n_i, f_i)

        w, h = f_i.shape[1], f_i.shape[0]
        for line in f_a_lines:
            line = line.split(',')
            label = int(line[-3])
            center_x = (float(line[0]) + float(line[2])/2)/w
            center_y = (float(line[1]) + float(line[3])/2)/h
            width = (float(line[2]))/w
            height = (float(line[3]))/h

            if label == 1 or label == 2:
                label = 0
            elif label == 9:
                label = 1
            elif label == 4 or label == 5:
                label = 2
            elif label == 6:
                label = 3
            elif label == 3 or label == 7 or label == 8 or label == 10:
                label = 4
            else:
                label = 12

            new_line = str(label) + " " + str(round(center_x, 4)) + " " + str(round(center_y,4)) + " " + str(round(width,4)) + " " + str(round(height,4)) + '\n'

            if label in [0, 1, 2, 3, 4]:
            #changelabelonly     f_a_l.write(new_line)
                if width > 0.01 and height > 0.01:
                    f_a_ls.write(new_line)

        f_a.close()
        # f_a_l.close()
        # f_a_ls.close()

print("Done")
