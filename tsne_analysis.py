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

def plot_point(layer_num):
    points = np.load('./save/' + str(layer_num) + '/tsne_point.npy')
    labels = np.load('./save/' + str(layer_num) + '/label.npy')
    numbers = list(range(len(labels)))

    labels_name = ['person', 'bus', 'car', 'truck', 'bike']
    fig = plt.figure(figsize=(80,80))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, s=10, edgecolors=None)
    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(handles, labels_name)

    for i, num in enumerate(numbers):
        plt.annotate(num, (points[:, 0][i], points[:, 1][i]), fontsize=3)

    plt.savefig('./save/' + str(layer_num) + '/annotate.png')

    xy_person = []
    xy_bus = []
    xy_car = []
    xy_truck = []
    xy_bike = []

    for i in range(len(labels)):
        if labels[i] == 0.0:
            xy_person.append(points[i])
        if labels[i] == 1.0:
            xy_bus.append(points[i])
        if labels[i] == 2.0:
            xy_car.append(points[i])
        if labels[i] == 3.0:
            xy_truck.append(points[i])
        if labels[i] == 4.0:
            xy_bike.append(points[i])

    fig = plt.figure(figsize=(50, 50))
    plt.subplot(3, 2, 1)
    plt.scatter(np.array(xy_person)[:, 0], np.array(xy_person)[:, 1], label='person', c='indigo', s=10, edgecolors='none')
    plt.title('person', fontsize=60)
    plt.subplot(3, 2, 2)
    plt.scatter(np.array(xy_bus)[:, 0], np.array(xy_bus)[:, 1], label='bus', s=10, c='mediumblue', edgecolors='none')
    plt.title('bus', fontsize=60)
    plt.subplot(3, 2, 3)
    plt.scatter(np.array(xy_car)[:, 0], np.array(xy_car)[:, 1], label='car', s=10, c='teal', edgecolors='none')
    plt.title('car', fontsize=60)
    plt.subplot(3, 2, 4)
    plt.scatter(np.array(xy_truck)[:, 0], np.array(xy_truck)[:, 1], label='truck', s=10, c='limegreen',
                edgecolors='none')
    plt.title('truck', fontsize=60)
    plt.subplot(3, 2, 5)
    plt.scatter(np.array(xy_bike)[:, 0], np.array(xy_bike)[:, 1], label='bike', s=10, c='orange',
                edgecolors='none')
    plt.title('bike', fontsize=60)

    plt.savefig('./save/' + str(layer_num) + '/class.png')
    print('d')

plot_point(17)
plot_point(20)
# plot_point(23)


























