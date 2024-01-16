import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import time
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def auto_downsample_ratio(h, w):
    return min(512/max(h, w), 1)

transform = transforms.Compose([
    transforms.Resize((360, 360)), 
    transforms.ToTensor()
])

model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")
model = model.eval()
model = model.to(device)

cap = cv2.VideoCapture("./3_0.mp4")
downsample_ratio = 1
rec = [None]*4

transform = A.Compose([
    A.Resize(360, 360),
    ToTensorV2()
])

bgr = torch.tensor([120, 255, 155], device=device, dtype=torch.float32).div(255).view(1, 1, 3, 1, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        transformed = frame.copy()[:, :, ::-1].astype("float32")/255.
        transformed = transform(image=transformed)["image"][None, ...].to(device)
        fgr, pha, *rec = model(transformed, *rec, 1)
        com = fgr*pha + bgr*(1-pha) # Shape: 1 x 1 x channels x height x width
        break
    break

