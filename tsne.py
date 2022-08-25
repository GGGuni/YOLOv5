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
import pytorch_model_summary

from models.yolo import Model
from utils.plots import feature_visualization


original_model = torch.load('./runs/train/exp53/weights/best.pt')['model'].float().cuda()
#print(original_model)

#model = nn.Sequential(original_model.model[:-1])
#model.model = model.model[:-1]

#pytorch_model_summary.summary(original_model, torch.zeros(1,4,1024,1024).cuda(), show_input=True, show_hierarchical=True, print_summary=True)



class NewModel(nn.Module):
    def __init__(self, output_layers, *args):
        super().__init__(*args)
        self.output_layers = output_layers
        # print(self.output_layers)
        self.selected_out = OrderedDict()
        # PRETRAINED MODEL
        self.pretrained = original_model.model
        self.fhooks = []

        for i, l in enumerate(list(self.pretrained._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.pretrained, l).register_forward_hook(self.forward_hook(l)))

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output

        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out

model = NewModel(output_layers=[22, 23]).cuda()


# def get_features(x, model, target_layer):
#     features = []
#     for i in range(0, 12):
#         x = (model.model[i])(x)
#
#         # if i == target_layer:
#         #     features[i] = x
#     return x

img_list = os.listdir('./data/test/images/test/')

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

    #feature = model(tensor)

    out, layerout = model(tensor)

    #feature = get_features(tensor, model, 11)


    #feature_visualization(feature, 'Focus', stage=0, n=32, save_dir=Path('runs/feature/exp'))


print(layerout)

