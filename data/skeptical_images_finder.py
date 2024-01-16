import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from  bisenet import BiSeNet
import torch
from torch import nn, optim
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import shutil
import os

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BiSeNet(2).eval()
model.activate_evaluation_mode()
model.to(device)
model.load_state_dict(torch.load("/kaggle/input/bisenet-trial-12-2/last.pt", map_location=device))

transform = A.Compose([A.Resize(360, 360), 
                      ToTensorV2()])
def preprocess_image(image_rgb):
    cloned = image_rgb.copy()
    cloned = image_rgb.astype("float32")/255.
    cloned = transform(image=cloned)["image"][None, ...].to(device)
    return cloned

img_dirs = glob.glob("/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images/*")
print(f"Num images: {len(img_dirs)}")

for img_dir in img_dirs:
    image_rgb = cv2.imread("/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images/1000092795.jpg")[:, :, ::-1]
    predictor.set_image(image_rgb)
    transformed = preprocess_image(image_rgb)
    pred = model(transformed)[0].argmax(0).cpu().numpy().astype("uint8")

    break
