from bisenet import BiSeNet
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch import nn
import random
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

torch.set_grad_enabled(False)

model = BiSeNet(2)
model.eval()
model.activate_evaluation_mode()
model.load_state_dict(torch.load("./bisenet_360_trial_7/last.pt", map_location="cpu"))

img_dirs = sorted(glob.glob("/home/edmond/Desktop/instance-level-human-parsing/images/*"))
mask_dirs = sorted(glob.glob("/home/edmond/Desktop/instance-level-human-parsing/masks/*"))

transform = A.Compose([A.Resize(360, 360), 
                      ToTensorV2()])

ious = []
for ith, img_dir in enumerate(img_dirs[1:]):
    image = cv2.imread(img_dir).astype("float32")/255.
    mask = cv2.imread(mask_dirs[ith+1], cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask==2, 1, 0)
    mask = cv2.resize(mask.astype("uint8"), (360, 360))
    transformed = transform(image=image)["image"][None, ...]
    pred = model(transformed).argmax(1)[0]
    iou = 2*(pred*mask).sum()/(pred.sum() + mask.sum())
    ious.append([img_dir, iou])

