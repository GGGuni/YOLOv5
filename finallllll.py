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

#pytorch_model_summary.summary(original_model, torch.zeros(1,4,1280,1280).cuda(), show_hierarchical=True,show_input=True, print_summary=True)



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



def get_new_indices(indices, cand):
    new_indices = []
    result = []
    for i in range(len(indices)):
        for k in range(3):
            if cand[..., 4:5][:, k, indices[i, 0], indices[i, 1]] > 0.9:
                new_indices.append([int(indices[i,0]), int(indices[i,1])])
                result = list(set(map(tuple, new_indices)))

    # change to list
    n_indices = []
    for j in range(len(result)):
        n_indices.append(list(result[j]))

    # return n_indices
    return torch.tensor(n_indices).reshape(-1,2)

def get_correct_indices(indices, cand, check_labels):
    correct_indices = []

    for i in range(len(indices)):
            for k in range(3):
                index = np.argmax(cand[...,5:10][:, k, indices[i][0], indices[i][1]].cpu())

                if int(index) == check_labels[i]:
                    correct_indices.append([int(indices[i,0]), int(indices[i,1])])

    result = list(set(map(tuple, correct_indices)))

    f_indices = []
    for j in range(len(result)):
        f_indices.append(list(result[j]))

    return torch.tensor(correct_indices).reshape(-1,2)

def get_features_labels(cand, tbox_gt, labels):
    check_labels = []
    new_indices = []

    new_cand = xywh2xyxy(cand[..., :4].reshape(-1, 4)).reshape(cand.shape[0], cand.shape[1], cand.shape[2], cand.shape[3], 4)
    for i in range(len(tbox_gt)):
        indices = cal_IOU_wFeatureMap(new_cand, torch.tensor(tbox_gt[i]).cuda())
        new_index = get_new_indices(indices, cand)
        if len(new_index) == 0:
            continue
        else:
            new_indices.append(new_index)

        for k in range(len(new_index)):
            check_labels.append(labels[i][0])

    # new_indices_1 = torch.tensor(new_indices_1).reshape(-1,2)

    if len(new_indices)==0:
        new_indices = []
    else:
        new_indices = torch.cat(new_indices, 0)

    final_indices = get_correct_indices(new_indices, cand, check_labels)


    return final_indices

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
        answer = labels[:, 0]

    features = list(visualization.values())

    final_indices_1 = get_features_labels(cand_1, tbox_gt, labels)
    for i in range(len(tbox_gt)):
        for j in range(len(final_indices_1)):
            s_features_17.append(features[17][..., final_indices_1[j][0], final_indices_1[j][1]].cpu())
            s_labels_17.append(labels[i][0])

    final_indices_2 = get_features_labels(cand_2, tbox_gt, labels)
    for i in range(len(tbox_gt)):
        for j in range(len(final_indices_2)):
            s_features_20.append(features[20][..., final_indices_2[j][0], final_indices_2[j][1]].cpu())
            s_labels_20.append(labels[i][0])

    final_indices_3 = get_features_labels(cand_3, tbox_gt, labels)
    for i in range(len(tbox_gt)):
        for j in range(len(final_indices_3)):
            s_features_23.append(features[23][..., final_indices_3[j][0], final_indices_3[j][1]].cpu())
            s_labels_23.append(labels[i][0])

    print('done')


for k in range(len(s_features_17)):
    f_cpu = s_features_17[k].cpu()
    np_features_17.append(np.array(f_cpu.squeeze(0)))

for k in range(len(s_features_20)):
    f_cpu = s_features_20[k].cpu()
    np_features_20.append(np.array(f_cpu.squeeze(0)))

for k in range(len(s_features_23)):
    f_cpu = s_features_23[k].cpu()
    np_features_23.append(np.array(f_cpu.squeeze(0)))

def plot_image(tsne, s_labels, layer_num):
    xy_person = []
    xy_bus = []
    xy_car = []
    xy_truck = []
    xy_bike = []

    for i in range(len(s_labels)):
        if s_labels[i] == 0.0:
            xy_person.append(tsne[i])
        if s_labels[i] == 1.0:
            xy_bus.append(tsne[i])
        if s_labels[i] == 2.0:
            xy_car.append(tsne[i])
        if s_labels[i] == 3.0:
            xy_truck.append(tsne[i])
        if s_labels[i] == 4.0:
            xy_bike.append(tsne[i])

    plt.figure()
    plt.scatter(np.array(xy_person)[:, 0], np.array(xy_person)[:, 1], label='person', s=0.5, edgecolors='none')
    plt.scatter(np.array(xy_bus)[:, 0], np.array(xy_bus)[:, 1], label='bus', s=0.5, edgecolors='none')
    plt.scatter(np.array(xy_car)[:, 0], np.array(xy_car)[:, 1], label='car', s=0.5, edgecolors='none')
    plt.scatter(np.array(xy_truck)[:, 0], np.array(xy_truck)[:, 1], label='truck', s=0.5, edgecolors='none')
    plt.scatter(np.array(xy_bike)[:, 0], np.array(xy_bike)[:, 1], label='bike', s=0.5, edgecolors='none')
    plt.legend()
    plt.savefig('./runs/tsne/'+str(layer_num)+'/all_'+str(layer_num)+'24.png')
    plt.subplot(3, 2, 1)
    plt.scatter(np.array(xy_person)[:, 0], np.array(xy_person)[:, 1], label='person', s=0.5, edgecolors='none')
    plt.title('person')
    plt.subplot(3, 2, 2)
    plt.scatter(np.array(xy_bus)[:, 0], np.array(xy_bus)[:, 1], label='bus', s=0.5, c='orange', edgecolors='none')
    plt.title('bus')
    plt.subplot(3, 2, 3)
    plt.scatter(np.array(xy_car)[:, 0], np.array(xy_car)[:, 1], label='car', s=0.5, c='green', edgecolors='none')
    plt.title('car')
    plt.subplot(3, 2, 4)
    plt.scatter(np.array(xy_truck)[:, 0], np.array(xy_truck)[:, 1], label='truck', s=0.5, c='red',
                edgecolors='none')
    plt.title('truck')
    plt.subplot(3, 2, 5)
    plt.scatter(np.array(xy_bike)[:, 0], np.array(xy_bike)[:, 1], label='bike', s=0.5, c='mediumpurple',
                edgecolors='none')
    plt.title('bike')
    plt.savefig('./runs/tsne/'+str(layer_num)+'/class24.png')


model_17 = TSNE(n_components=2, learning_rate=200.0, n_iter=10000)
tsne_17 = model_17.fit_transform(np_features_17)
plot_image(tsne_17, s_labels_17, 17)

model_20 = TSNE(n_components=2, learning_rate=200.0, n_iter=10000)
tsne_20 = model_20.fit_transform(np_features_20)
plot_image(tsne_20, s_labels_20, 20)

model_23 = TSNE(n_components=2, learning_rate=200.0, n_iter=10000)
tsne_23 = model_23.fit_transform(np_features_23)
plot_image(tsne_23, s_labels_23, 23)













