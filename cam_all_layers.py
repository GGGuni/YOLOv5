import os
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import cv2
import numpy as np
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
from utils.general import non_max_suppression

COLORS = np.random.uniform(0, 255, size=(80, 3))

def parse_detections(results):
    detections = results.pandas().xyxy[0]
    detections = detections.to_dict()
    boxes, colors, names = [], [], []

    for i in range(len(detections["xmin"])):
        confidence = detections["confidence"][i]
        if confidence < 0.2:
            continue
        xmin = int(detections["xmin"][i])
        ymin = int(detections["ymin"][i])
        xmax = int(detections["xmax"][i])
        ymax = int(detections["ymax"][i])
        name = detections["name"][i]
        category = int(detections["class"][i])
        color = COLORS[category]

        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append(name)
    return boxes, colors, names


def draw_detections(boxes, colors, names, img):
    label = ['person', 'bus', 'car', 'truck', 'bike']
    for box, color, name in zip(boxes, colors, names):
        color = np.array([255*color, 255*color, 255*color])
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(
            img,
            (int(xmin), int(ymin)),
            (int(xmax), int(ymax)),
            color,
            2)

        cv2.putText(img, label[int(name)], (int(xmin), int(ymin) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return img


model = torch.load('./runs/train/exp53/weights/best.pt')['model'].float().cuda()

target_layers_list = [[model.model[6]], [model.model[9]], [model.model[13]], [model.model[17]], [model.model[20]], [model.model[23]]]

img_file = "./data/test/images/test/0000006_04050_d_0000010.jpg"
edge_file = "./data/test/edges/test/0000006_04050_d_0000010.png"

img = np.array(Image.open(img_file))
img = cv2.resize(img, (1280, 1280))
rgb_img = img.copy()
img = np.float32(img) / 255

edge_img = np.array(Image.open(edge_file))
edge_img = cv2.resize(edge_img, (1280, 1280))
edge_rgb_img = edge_img.copy()
edge_img = np.float32(edge_img) / 255

transform = transforms.ToTensor()
tensor = transform(np.concatenate((edge_img.reshape(1280,1280,1), img),axis=-1)).unsqueeze(0).cuda()
#tensor = transform(np.concatenate((img, edge_img.reshape(1280,1280,1)),axis=-1)[:,:,::-1].copy()).unsqueeze(0).cuda()

results = model(tensor)

def parse_boxes(results):
    boxes = results[:,:4].cpu().numpy()
    conf = results[:,4].cpu().numpy()
    names = results[:,5].cpu().numpy()
    return boxes, conf, names

out = non_max_suppression(results[0], 0.1, 0.5, labels=[], multi_label=True, agnostic=False)
boxes, colors, names = parse_boxes(out[0])
detections = draw_detections(boxes, colors, names, rgb_img.copy())
Image.fromarray(detections)

for lc, target_layers in enumerate(target_layers_list):
    cam = EigenCAM(model, target_layers, use_cuda=True)
    grayscale_cam = cam(tensor)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    Image.fromarray(cam_image)

    def renormalize_cam_in_bounding_boxes(boxes, colors, names, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1]
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            if x1 < 0:
                x1 = 0
            if x2 < 0:
                x2 = 0
            if y1 < 0:
                y1 = 0
            if y2 < 0:
                y2 = 0

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            #renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[x1:x2, y1:y2].copy())
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        image_with_bounding_boxes = draw_detections(boxes, colors, names, eigencam_image_renormalized)
        return image_with_bounding_boxes


    renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img[...,::-1], grayscale_cam)
    Image.fromarray(renormalized_cam_image)

    cv2.imwrite("./cam_all/"+str(lc)+".jpg", np.hstack((rgb_img[...,::-1], cam_image, renormalized_cam_image)))
    #cv2.imshow("asdf", np.hstack((rgb_img[...,::-1], cam_image, renormalized_cam_image)))
    # cv2.waitKey(0)
    print("1 step done")
