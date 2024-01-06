import glob
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
from tqdm import tqdm

img_dirs = sorted(glob.glob("./images/*"))
mask_dirs = sorted(glob.glob("./masks/*"))
human_id_dirs = sorted(glob.glob("./Human_ids/*"))

aux_img_dirs = glob.glob("./distraction_images/*")
print(f"Num images: {len(img_dirs)}, Num masks: {len(mask_dirs)}, Num Human Id Images: {len(human_id_dirs)}, Num Aux Images: {len(aux_img_dirs)}")


class CreateFineTuningDataset(Dataset):
    def __init__(self, img_dirs, mask_dirs, aux_img_dirs, human_id_dirs, transform, val_set = False):
        self.img_dirs = img_dirs
        self.mask_dirs = mask_dirs
        self.aux_img_dirs = aux_img_dirs
        self.human_id_dirs = human_id_dirs
        self.transform = transform = transform
        self.val_set = val_set
        
    def __len__(self):
        return len(self.img_dirs)
    
    @staticmethod
    def convertColorChannel(img_dir):
        return cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB)
        
    @staticmethod
    def matte_images(primary_image, secondary_image):
        '''
        primary_image: The background image
        secondary_image: The single extracted person image
        '''
        primary_image = cv2.resize(primary_image, (secondary_image.shape[1], secondary_image.shape[0]))
        primary_image_mask = np.where(secondary_image.mean(-1) > 0, 0, 1)[..., None]
        primary_image_mask = np.tile(primary_image_mask, reps = (1, 1, 3))
        combination_image = primary_image*primary_image_mask + secondary_image
        return combination_image
    
    @staticmethod
    def extractPerson(image, human_ids_image):
        person = np.zeros_like(image)
        num_persons = len(np.unique(human_ids_image))
        person = np.where(human_ids_image==random.randint(1, num_persons), 1, 0)[..., None]
        person = np.tile(person, reps = (1, 1, 3))
        person = person*image
        return person

    def __getitem__(self, idx):
        img_dir = self.img_dirs[idx]
        mask_dir = self.mask_dirs[idx]
        image = self.convertColorChannel(img_dir)
        
        if not self.val_set:
            
            mask = np.where(cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)==2, 1, 0)
#             if random.randint(0, 10) >= 0:
            human_ids_image = cv2.imread(human_id_dirs[idx], cv2.IMREAD_GRAYSCALE)
            person = self.extractPerson(image, human_ids_image)
            
            sub_mask = np.where(person.mean(-1) > 0, 1, 0)
            sub_mask_area_ratio = sub_mask.sum()/sub_mask.size

            if sub_mask_area_ratio > 0.15:
                print(sub_mask_area_ratio)
                mask = np.where(person.mean(-1) > 0, mask, 0)
                aux_image = self.convertColorChannel(random.choice(self.aux_img_dirs))
                image = self.matte_images(aux_image, person)
        else:
            mask = np.where(cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE) > 100, 1, 0)
            
        image = image.astype("float32")/255.0
        augmented = self.transform(image=image, mask = mask)
        image = augmented["image"]
        mask = augmented["mask"].long()
        return image, mask

transform = A.Compose([A.Resize(448, 448),
                       ToTensorV2()])
train_dataset = CreateFineTuningDataset(img_dirs, mask_dirs, aux_img_dirs, human_id_dirs, transform, val_set = False)

val_img_dirs = sorted(glob.glob("../../hair-seg.v6i.coco-segmentation/roboflow_val_set/train/*"))
val_mask_dirs = sorted(glob.glob("../../hair-seg.v6i.coco-segmentation/roboflow_val_set/train_masks//*"))
print(f"Num val Images: {len(val_img_dirs)}, Num val masks: {len(val_mask_dirs)}")
val_dataset = CreateFineTuningDataset(val_img_dirs, val_mask_dirs, None, None, transform, True)

train_img_dirs = sorted(glob.glob("./images/*"))
train_mask_dirs = sorted(glob.glob("./masks/*"))
aux_img_dirs = glob.glob("./distraction_images/*")
human_ids_dirs = sorted(glob.glob("./Human_ids/*"))

print(f"Num val Images: {len(train_img_dirs)}, Num val masks: {len(train_mask_dirs)}, Num human ids: {len(human_ids_dirs)}")
train_dataset = CreateFineTuningDataset(train_img_dirs, train_mask_dirs, aux_img_dirs, human_ids_dirs, transform, False)

