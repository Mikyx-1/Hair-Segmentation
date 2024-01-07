                                                                                             
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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BiSeNet(2)
model.eval()
model.to(device)
model.activate_evaluation_mode()
model.load_state_dict(torch.load("./bisenet_360_trial_7/val_loss_min.pt", map_location="cpu"))

img_dirs = sorted(glob.glob("../train_dataset/images/*"))
mask_dirs = sorted(glob.glob("../train_dataset/masks/*"))
print(f"Num images: {len(img_dirs)}, Num masks: {len(mask_dirs)}")


transform = A.Compose([A.Resize(360, 360), 
                      ToTensorV2()])

ious = []
for i in tqdm(range(len(img_dirs))):
    image = cv2.imread(img_dirs[i]).astype("float32")/255.
    mask = cv2.imread(mask_dirs[i], cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask==2, 1, 0)
    mask = cv2.resize(mask.astype("uint8"), (360, 360))
    transformed = transform(image=image)["image"][None, ...]
    pred = model(transformed).argmax(1)[0]
    iou = 2*(pred*mask + 0.0001).sum()/(pred.sum() + mask.sum() + 0.0001)
    ious.append([img_dir, iou])

sorted_ious = sorted(ious, key=lambda x: x[1])
with open("sorted_ious.txt", "w") as file:
  for sorted_iou in tqdm(sorted_ious):
    file.write(sorted_iou[0] + " " + str(sorted_iou[1]) + "\n")

