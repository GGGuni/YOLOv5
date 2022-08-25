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
import random
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


np_features_17 = []
np_features_20 = []
np_features_23 = []

s_features_17 = []
s_labels_17 = []
s_gt_17 = []
s_image_17 = []
s_features_20 = []
s_labels_20 = []
s_gt_20 = []
s_image_20 = []
s_features_23 = []
s_labels_23 = []
s_gt_23 = []
s_image_23 = []

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
        tbox_gt_origin = xywh2xyxy(labels[:, 1:5])

    new_cand_1 = xywh2xyxy(cand_1[..., :4].reshape(-1, 4)).reshape(cand_1.shape[0], cand_1.shape[1], cand_1.shape[2], cand_1.shape[3], 4)
    new_cand_2 = xywh2xyxy(cand_2[..., :4].reshape(-1, 4)).reshape(cand_2.shape[0], cand_2.shape[1], cand_2.shape[2], cand_2.shape[3], 4)
    new_cand_3 = xywh2xyxy(cand_3[..., :4].reshape(-1, 4)).reshape(cand_3.shape[0], cand_3.shape[1], cand_3.shape[2], cand_3.shape[3], 4)

    features = list(visualization.values())

    for i in range(len(tbox_gt)):
        indices = cal_IOU_wFeatureMap(new_cand_1, torch.tensor(tbox_gt[i]).cuda())
        for j in range(len(indices)):
            s_features_17.append(features[17][..., indices[j][0], indices[j][1]].cpu())
            s_labels_17.append(labels[i][0])
            s_gt_17.append(tbox_gt_origin[i])
            s_image_17.append(image_full_path)

    for i in range(len(tbox_gt)):
        indices = cal_IOU_wFeatureMap(new_cand_2, torch.tensor(tbox_gt[i]).cuda())
        for j in range(len(indices)):
            s_features_20.append(features[20][..., indices[j][0], indices[j][1]].cpu())
            s_labels_20.append(labels[i][0])
            s_gt_20.append(tbox_gt_origin[i])
            s_image_20.append(image_full_path)

    for i in range(len(tbox_gt)):
        indices = cal_IOU_wFeatureMap(new_cand_3, torch.tensor(tbox_gt[i]).cuda())
        for j in range(len(indices)):
            s_features_23.append(features[23][..., indices[j][0], indices[j][1]].cpu())
            s_labels_23.append(labels[i][0])
            s_gt_23.append(tbox_gt_origin[i])
            s_image_23.append(image_full_path)

    print('done')

np.save('./save/17/label', s_labels_17)
np.save('./save/20/label', s_labels_20)
np.save('./save/23/label', s_labels_23)

def save_crop(s_gt, s_image, layer_num):
    for i in range(len(s_image)):
        img = Image.open(s_image[i])
        crop = img.crop((s_gt[i][0] * list(img.size)[0], s_gt[i][1] * list(img.size)[1],
                             s_gt[i][2] * list(img.size)[0], s_gt[i][3] * list(img.size)[1]))

        crop.save('./save/' + str(layer_num) + '/crop/'+str(i)+'.png')

for k in range(len(s_features_17)):
    f_cpu = s_features_17[k].cpu()
    np_features_17.append(np.array(f_cpu.squeeze(0)))
np.save('./save/17/feature', np_features_17)

for k in range(len(s_features_20)):
    f_cpu = s_features_20[k].cpu()
    np_features_20.append(np.array(f_cpu.squeeze(0)))
np.save('./save/20/feature', np_features_20)

for k in range(len(s_features_23)):
    f_cpu = s_features_23[k].cpu()
    np_features_23.append(np.array(f_cpu.squeeze(0)))
np.save('./save/23/feature', np_features_23)

model_17 = TSNE(n_components=2, learning_rate=200.0, n_iter=10000)
tsne_17 = model_17.fit_transform(np_features_17)
np.save('./save/17/tsne_point', tsne_17)
save_crop(s_gt_17, s_image_17, 17)

model_20 = TSNE(n_components=2, learning_rate=200.0, n_iter=10000)
tsne_20 = model_20.fit_transform(np_features_20)
np.save('./save/20/tsne_point', tsne_20)
save_crop(s_gt_20, s_image_20, 20)

model_23 = TSNE(n_components=2, learning_rate=200.0, n_iter=10000)
tsne_23 = model_23.fit_transform(np_features_23)
np.save('./save/23/tsne_point', tsne_23)
save_crop(s_gt_23, s_image_23, 23)
print('d')

def plot_random_crop(s_labels, s_features, s_gt, s_image, layer_num):
    xy_person = []
    xy_bus = []
    xy_car = []
    xy_truck = []
    xy_bike = []
    np_features = []
    for k in range(len(s_features)):
        f_cpu = s_features[k].cpu()
        np_features.append(np.array(f_cpu.squeeze(0)))

    model = TSNE(n_components=2, learning_rate=200.0, n_iter=10000)
    tsne = model.fit_transform(np_features)

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

    random_index = []

    for i in range(5):
        random_index.append(random.randint(0, len(s_features) - 1))

    np_features =[]
    rand_features = []
    rand_labels = []
    rand_gt = []
    rand_img = []

    for x in random_index:
        rand_features.append(s_features[x])
        rand_labels.append(s_labels[x])
        rand_gt.append(s_gt[x])
        rand_img.append(s_image[x])

    for k in range(len(rand_features)):
        f_cpu = rand_features[k].cpu()
        np_features.append(np.array(f_cpu.squeeze(0)))


    plt.figure()
    plt.scatter(np.array(xy_person)[:, 0], np.array(xy_person)[:, 1], s=10)
    plt.savefig('./runs/tsne/'+str(layer_num)+'/random/person.png')

    plt.figure()
    plt.scatter(np.array(xy_bus)[:, 0], np.array(xy_bus)[:, 1], s=10)
    plt.savefig('./runs/tsne/' + str(layer_num) + '/random/bus.png')
    plt.figure()
    plt.scatter(np.array(xy_car)[:, 0], np.array(xy_car)[:, 1], s=10)
    plt.savefig('./runs/tsne/' + str(layer_num) + '/random/car.png')
    plt.figure()
    plt.scatter(np.array(xy_truck)[:, 0], np.array(xy_truck)[:, 1], s=10)
    plt.savefig('./runs/tsne/' + str(layer_num) + '/random/truck.png')
    plt.figure()
    plt.scatter(np.array(xy_bike)[:, 0], np.array(xy_bike)[:, 1], s=10)
    plt.savefig('./runs/tsne/' + str(layer_num) + '/random/bike.png')

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    img_1 = Image.open(rand_img[0])
    crop_1 = img_1.crop((rand_gt[0][0] * list(img_1.size)[0], rand_gt[0][1] * list(img_1.size)[1],
                         rand_gt[0][2] * list(img_1.size)[0], rand_gt[0][3] * list(img_1.size)[1]))
    ax1.imshow(crop_1)

    ax2 = fig.add_subplot(2, 3, 2)
    img_2 = Image.open(rand_img[1])
    crop_2 = img_2.crop((rand_gt[1][0] * list(img_2.size)[0], rand_gt[1][1] * list(img_2.size)[1],
                         rand_gt[1][2] * list(img_2.size)[0], rand_gt[1][3] * list(img_2.size)[1]))
    ax2.imshow(crop_2)

    ax3 = fig.add_subplot(2, 3, 3)
    img_3 = Image.open(rand_img[2])
    crop_3 = img_3.crop((rand_gt[2][0] * list(img_3.size)[0], rand_gt[2][1] * list(img_3.size)[1],
                         rand_gt[2][2] * list(img_3.size)[0], rand_gt[2][3] * list(img_3.size)[1]))
    ax3.imshow(crop_3)

    ax4 = fig.add_subplot(2, 3, 4)
    img_4 = Image.open(rand_img[3])
    crop_4 = img_4.crop((rand_gt[3][0] * list(img_4.size)[0], rand_gt[3][1] * list(img_4.size)[1],
                         rand_gt[3][2] * list(img_4.size)[0], rand_gt[3][3] * list(img_4.size)[1]))
    ax4.imshow(crop_4)

    ax5 = fig.add_subplot(2, 3, 5)
    img_5 = Image.open(rand_img[4])
    crop_5 = img_5.crop((rand_gt[4][0] * list(img_5.size)[0], rand_gt[4][1] * list(img_5.size)[1],
                         rand_gt[4][2] * list(img_5.size)[0], rand_gt[4][3] * list(img_5.size)[1]))
    ax5.imshow(crop_5)
    plt.savefig('./runs/tsne/'+str(layer_num)+'/crop/img1.png')





# for k in range(len(s_features_17)):
#     f_cpu = s_features_17[k].cpu()
#     np_features_17.append(np.array(f_cpu.squeeze(0)))
#
# for k in range(len(s_features_20)):
#     f_cpu = s_features_20[k].cpu()
#     np_features_20.append(np.array(f_cpu.squeeze(0)))

# for k in range(len(s_features_23)):
#     f_cpu = s_features_23[k].cpu()
#     np_features_23.append(np.array(f_cpu.squeeze(0)))



# model_17 = TSNE(n_components=2, learning_rate=200.0, n_iter=10000)
# tsne_17 = model_17.fit_transform(np_features_17)
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


# fig = plt.figure()
# plt.scatter(tsne_23[0, 0], tsne_23[0, 1], label=rand_labels_23[0], s=10, c='red')
# plt.scatter(tsne_23[1, 0], tsne_23[1, 1], label=rand_labels_23[1], s=10, c='orange')
# plt.scatter(tsne_23[2, 0], tsne_23[2, 1], label=rand_labels_23[2], s=10, c='yellow')
# plt.scatter(tsne_23[3, 0], tsne_23[3, 1], label=rand_labels_23[3], s=10, c='green')
# plt.scatter(tsne_23[4, 0], tsne_23[4, 1], label=rand_labels_23[4], s=10, c='blue')

# plt.legend()
# plt.savefig('./runs/tsne/23/random1.png')

# plot_random_crop(s_labels_23, s_features_23, s_gt_23, s_image_23, 23)

















