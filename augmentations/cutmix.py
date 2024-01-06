import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random

aux_img_dirs = glob.glob("./distraction_images/*")
train_img_dirs = sorted(glob.glob("./fine_tuning_dataset/images/*"))
train_mask_dirs = sorted(glob.glob("./fine_tuning_dataset/masks/*"))
print(f"Num Images: {len(train_img_dirs)}, Num Masks: {len(train_mask_dirs)}, Num Auxiliary Images: {len(aux_img_dirs)}")

class CreateDataset(Dataset):
    def __init__(self, img_dirs, mask_dirs, aux_img_dirs=None, transform=transform):
        self.aux_img_dirs = aux_img_dirs
        self.transform = transform
        self.img_dirs = img_dirs
        self.mask_dirs = mask_dirs
        
    def __len__(self):
        return len(self.img_dirs)
    
    def __getitem__(self, idx):
        img_dir = self.img_dirs[idx]
        mask_dir = self.mask_dirs[idx]
        image = cv2.resize(cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB), (448, 448))
        mask = cv2.resize(np.where(cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)==2, 1, 0).astype("float32"), (448, 448))
       
        
        if self.aux_img_dirs is not None and random.randint(0, 10) > 7:
            x0 = random.randint(0, 224)
            y0 = random.randint(0, 224)
            aux_img_dir = random.choice(self.aux_img_dirs)
            aux_image = cv2.resize(cv2.cvtColor(cv2.imread(aux_img_dir), cv2.COLOR_BGR2RGB), (224, 224))
            image[y0:y0+224, x0:x0+224, :] = aux_image
            mask[y0:y0+224, x0:x0+224] = 0
        image = image.astype("float32")/255.0
        augmented = transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
        return image, mask

ds = CreateDataset(train_img_dirs, train_mask_dirs, aux_img_dirs=aux_img_dirs, transform=transform)
