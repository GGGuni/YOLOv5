from pathlib import Path

import sns as sns
import torch
import os
import torch.nn as nn
import torchvision.models as models
import cv2
import torchvision.transforms as transforms
import yaml
from sklearn.manifold import TSNE
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pytorch_model_summary

from yaml import SafeLoader

from models.yolo import Model
from utils.general import xywh2xyxy
from utils.plots import feature_visualization, plot_images
from iou import cal_IOU_wFeatureMap

original_model = torch.load('./runs/train/exp53/weights/best.pt')['model'].float().cuda()

# pytorch_model_summary.summary(original_model, torch.zeros(1,4,1280,1280).cuda(), show_hierarchical=True,show_input=True, print_summary=True)



# load anchors
# with open('./models/hub/anchors.yaml') as f:
#     anchor = yaml.load(f, SafeLoader)
#
# anchors = anchor["anchors_p5_640"]



visualization = {}

def hook_fn(m, i, o):
  visualization[m] = o

def get_all_layers(net):
  for name, layer in net.model._modules.items():
    if isinstance(layer, nn.Sequential):
      pass
    else:
      layer.register_forward_hook(hook_fn)


img_list = os.listdir('./data/test/images/test/')
label_list = os.listdir('./data/test/labels/test/')

tmp_list = []
output_list = []
layer_num = 23
feat_vect = []



def get_IoU(img_name, cand, features):
#cand[0,0], cand[0,1], cand[0,2] 모두 생각

    label_full_path = "./data/test/labels/test/" + img_name + 'txt'
    str_labels = []

    with open(label_full_path) as f:
        for line in f:
            str_labels.append(line.strip('\n').split(' '))

    labels = [list(map(float, x)) for x in str_labels]
    labels = np.array(labels)

    tbox_gt = xywh2xyxy(labels[:, 1:5]*1280)

    cand_cpu = cand.cpu()
    cand = np.array(cand_cpu)

    s_features = []
    s_labels = []



    # 좀더 깔끔하게 코드 짜는 것 고민해 보기
    # for j in range(160):
    #     for k in range(160):

    # for i in range(len(tbox_gt)):
    #     for j in range(160):
    #         for k in range(160):
    #             for l in range(3):
    #                 gt_area = (tbox_gt[:, 2][i] - tbox_gt[:, 0][i]) * (tbox_gt[:, 3][i] - tbox_gt[:, 1][i])

    for j in range(len(cand[0, 0])):
        for k in range(len(cand[0, 0, 0])):
            for l in range(3):
                print("Done j,k,l:", j,k,l)
                for i in range(len(tbox_gt)):
                    gt_area = (tbox_gt[:, 2][i] - tbox_gt[:, 0][i]) * (tbox_gt[:, 3][i] - tbox_gt[:, 1][i])

                    cand_xywh = cand[0, l, j, k, :4]
                    left_x = cand_xywh[0] - cand_xywh[2] / 2
                    left_y = cand_xywh[1] - cand_xywh[3] / 2
                    right_x = cand_xywh[0] + cand_xywh[2] / 2
                    right_y = cand_xywh[1] + cand_xywh[3] / 2

                    if left_x < 0:
                        left_x = 0
                    if left_y < 0:
                        left_y = 0

                    if right_x > 1280:
                        right_x = 1280
                    if right_y > 1280:
                        right_y = 1280

                    cand_area = (right_x - left_x) * (right_y - left_y)

                    x1 = max(tbox_gt[:, 0][i], left_x)
                    y1 = max(tbox_gt[:, 1][i], left_y)
                    x2 = min(tbox_gt[:, 2][i], right_x)
                    y2 = min(tbox_gt[:, 3][i], right_y)

                    w = max(0, x2-x1)
                    h = max(0, y2-y1)

                    # intersection = np.clip(w, 0, 1280) * np.clip(h, 0, 1280)
                    intersection = w*h
                    iou = intersection / (gt_area + cand_area - intersection)

                if iou > 0.5:
                    s_features.append(features[..., j, k])
                    s_labels.append(labels[:, 0][i])
                    break

    print('done')

    return s_features, s_labels


np_features_17 = []
np_features_20 = []
np_features_23 = []

s_features_17 = []
s_labels_17 = []
s_features_20 = []
s_labels_20 = []
s_features_23 = []
s_labels_23 = []

for img_path in img_list:

    image_full_path = "./data/test/images/test/"+img_path
    edge_image_full_path = "./data/test/edges/test/"+img_path[:-3]+'png'

    img = np.array(Image.open(image_full_path))
    img = cv2.resize(img, (1280, 1280))
    img = np.float32(img) / 255

    edge_img = np.array(Image.open(edge_image_full_path))
    edge_img = cv2.resize(edge_img, (1280, 1280))
    edge_img = np.float32(edge_img) / 255

    transform = transforms.ToTensor()
    tensor = transform(np.concatenate((edge_img.reshape(1280, 1280, 1), img), axis=-1)).unsqueeze(0).cuda()

    get_all_layers(original_model)
    out = original_model(tensor)

    """
    out[0][0:1, 0~size1] - link with features[17]
    out[0][0:1, size1~size2] - link with features[20]
    out[0][0:1, size2~size3] - link with features[23]
    """

    size1 = 3 * out[1][0].shape[2] * out[1][0].shape[3]
    size2 = size1 + 3 * out[1][1].shape[2] * out[1][1].shape[3]
    size3 = size2 + 3 * out[1][1].shape[2] * out[1][1].shape[3]

    cand_1 = out[0][0:1, 0:size1].view(1, 3, out[1][0].shape[2], out[1][0].shape[3], 10)[..., 0:10]
    cand_2 = out[0][0:1, size1:size2].view(1, 3, out[1][1].shape[2], out[1][1].shape[3], 10)[..., 0:10]
    cand_3 = out[0][0:1, size2:size3].view(1, 3, out[1][2].shape[2], out[1][2].shape[3], 10)[..., 0:10]

    cpu = cand_1.cpu()

    label_full_path = "./data/test/labels/test/" + img_path[:-3] + 'txt'

    str_labels = []

    with open(label_full_path) as f:
        for line in f:
            str_labels.append(line.strip('\n').split(' '))

    labels = [list(map(float, x)) for x in str_labels]
    labels = np.array(labels)

    if len(labels) <= 0:
        continue
    else:
        tbox_gt = xywh2xyxy(labels[:, 1:5] * 1280)

    new_cand_1 = xywh2xyxy(cand_1[..., :4].reshape(-1, 4)).reshape(cand_1.shape[0], cand_1.shape[1], cand_1.shape[2], cand_1.shape[3], 4)
    new_cand_2 = xywh2xyxy(cand_2[..., :4].reshape(-1, 4)).reshape(cand_2.shape[0], cand_2.shape[1], cand_2.shape[2], cand_2.shape[3], 4)
    new_cand_3 = xywh2xyxy(cand_3[..., :4].reshape(-1, 4)).reshape(cand_3.shape[0], cand_3.shape[1], cand_3.shape[2], cand_3.shape[3], 4)

    features = list(visualization.values())

    # for i in range(len(tbox_gt)):
    #     indices = cal_IOU_wFeatureMap(new_cand_1, torch.tensor(tbox_gt[i]).cuda())
    #     for j in range(len(indices)):
    #         s_features_17.append(features[17][..., indices[j][0], indices[j][1]].cpu())
    #         s_labels_17.append(labels[i][0])
    #
    # for i in range(len(tbox_gt)):
    #     indices = cal_IOU_wFeatureMap(new_cand_2, torch.tensor(tbox_gt[i]).cuda())
    #     for j in range(len(indices)):
    #         s_features_20.append(features[20][..., indices[j][0], indices[j][1]].cpu())
    #         s_labels_20.append(labels[i][0])

    for i in range(len(tbox_gt)):
        indices = cal_IOU_wFeatureMap(new_cand_3, torch.tensor(tbox_gt[i]).cuda())
        for j in range(len(indices)):
            s_features_23.append(features[23][..., indices[j][0], indices[j][1]].cpu())
            s_labels_23.append(labels[i][0])
    print('done')


# for k in range(len(s_features_17)):
#     f_cpu = s_features_17[k].cpu()
#     np_features_17.append(np.array(f_cpu.squeeze(0)))
#
# for k in range(len(s_features_20)):
#     f_cpu = s_features_20[k].cpu()
#     np_features_20.append(np.array(f_cpu.squeeze(0)))

for k in range(len(s_features_23)):
    f_cpu = s_features_23[k].cpu()
    np_features_23.append(np.array(f_cpu.squeeze(0)))

# model_17 = TSNE(n_components=2, learning_rate=200.0, n_iter=10000)
# tsne_17 = model_17.fit_transform(np_features_17)
s_labels_name = ['person', 'bus', 'car', 'truck', 'bike']
# fig = plt.figure()
# scatter = plt.scatter(tsne_17[:, 0], tsne_17[:, 1], c=s_labels_17, s=1)
# handles_17, _ = scatter.legend_elements(prop='colors')
# plt.legend(handles_17, s_labels_name)
# plt.savefig('./runs/tsne/17/image_17.png')
#
# model_20 = TSNE(n_components=2, learning_rate=200.0, n_iter=10000)
# tsne_20 = model_20.fit_transform(np_features_20)
# fig = plt.figure()
# scatter = plt.scatter(tsne_20[:, 0], tsne_20[:, 1], c=s_labels_20, s=1)
# handles_20, _20 = scatter.legend_elements(prop='colors')
# plt.legend(handles_20, s_labels_name)
# plt.savefig('./runs/tsne/20/image_20.png')

model_23 = TSNE(n_components=2, learning_rate=200.0, n_iter=10000)
tsne_23 = model_23.fit_transform(np_features_23)
fig = plt.figure()
scatter = plt.scatter(tsne_23[:, 0], tsne_23[:, 1], c=s_labels_23, s=1)
handles_23, _23 = scatter.legend_elements(prop='colors')
plt.legend(handles_23, s_labels_name)
plt.savefig('./runs/tsne/23/image_23.png')














