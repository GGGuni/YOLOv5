from collections import OrderedDict
from pathlib import Path

import torch
import os
import torch.nn as nn
import torchvision.models as models
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pytorch_model_summary
from torchvision.utils import save_image

from models.yolo import Model
from utils.plots import feature_visualization, plot_images

original_model = torch.load('./runs/train/exp53/weights/best.pt')['model'].float().cuda()
#print(original_model)

#model = nn.Sequential(original_model.model[:-1])
#model.model = model.model[:-1]

#pytorch_model_summary.summary(original_model, torch.zeros(1,4,1280,1280).cuda(), show_hierarchical=True,show_input=True, print_summary=True)

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

tmp_list = []
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

    features = list(visualization.values())
    # feat = features[23]
    # np_feat = feat.cpu().numpy()
    # for i in range(24):
    i=23

    np_feat = features[i].cpu().numpy()
    np.save('runs/features/all'+str(i), np_feat)
    # tmp_load = np.load('runs/features/all'+str(i)+'.npy')
    # np.abs(tmp_load - np_feat).max()
    # if i == 23:
    tmp_list.append(np_feat)

aggre = np.array(tmp_list)

all_img_l23_min = np.array(tmp_list).min()
all_img_l23_max = np.array(tmp_list).max()
all_img_l23_mean = np.array(tmp_list).mean()
all_img_l23_std = np.array(tmp_list).std()
all_img_l23_median = np.median(np.array(tmp_list))

print('min : ', all_img_l23_min)
print('max : ', all_img_l23_max)
print('mean : ', all_img_l23_mean)
print('std : ', all_img_l23_std)
print('median : ', all_img_l23_median)

print("Done")

