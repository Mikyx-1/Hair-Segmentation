from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CreateDataset(Dataset):
  def __init__(self, img_dirs, mask_dirs, transform):
    assert transform is not None, "Transform is None, can't proceed to the next step"
    self.transform = transform
    self.img_dirs = img_dirs
    self.mask_dirs = mask_dirs
    self.val_set = val_set

  def __len__(self):
    return len(self.img_dirs)

  def __getitem__(self, idx):
    img_dir = self.img_dirs[idx]
    mask_dir = self.mask_dirs[idx]
    image = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB).astype("float32")/255.0
    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)


    augmented = self.transform(image=image, mask = mask)
    image = augmented["image"]
    mask = augmented["mask"].long()
    return image, mask
  

def createDataLoader(img_dirs, mask_dirs, val_set, shuffle, batch_size, num_workers):
   '''
   Val set (Boolean): True == training, False == evaluation
   '''

   if not val_set:
    transform = A.Compose([A.Resize(360, 360),
                             A.Flip(p=0.45),
                             A.OneOf([
                               A.ChannelShuffle(p=0.5),
           
                               A.GaussNoise(var_limit = (10.0, 50.0), mean = 0, p=0.5)
                             ], p = 0.5),
                       ToTensorV2()])
   else:
    transform = A.Compose([A.Resize(360, 360),
                       ToTensorV2()])
    pin_memory = False
   if torch.cuda.is_available():
    pin_memory = True
   return DataLoader(CreateDataset(img_dirs, mask_dirs, transform), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

def createTestLoader(img_dirs, mask_dirs, batch_size, num_workers):
    transform = A.Compose([A.Resize(360, 360),
                       ToTensorV2()])
    pin_memory = True if torch.cuda.is_available() else False  
    return DataLoader(CreateDataset(img_dirs, mask_dirs, transform, val_set=False), shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
