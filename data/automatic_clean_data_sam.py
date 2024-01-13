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
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BiSeNet(2).eval()
model.activate_evaluation_mode()
model.to(device)
model.load_state_dict(torch.load("../last.pt", map_location=device))

sam = sam_model_registry["default"](checkpoint="./sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

def unified_decision(sam_masks, bise_mask):
    iou_max = 0.85
    iou_max_mask = None
    for sam_mask in sam_masks:
        sam_mask = sam_mask["segmentation"].astype("uint8")
        iou = (2*(bise_mask*sam_mask).sum() + 1e-6)/(bise_mask.sum() + sam_mask.sum() + 1e-6)
        if iou >= 0.85 and iou >= iou_max:
            iou_max_mask = sam_mask
            iou_max = iou
    return sam_mask
  

img_dirs = glob.glob("./video_images/*")
print(f"Num images: {len(img_dirs)}")


transform = A.Compose([A.Resize(360, 360), 
                      ToTensorV2()])


# perplexities = []
for i in tqdm(range(len(img_dirs))):
    image = cv2.imread(img_dirs[i])
    if image is not None:
        image = image[:, :, ::-1]
        sam_image = image.copy()
        image_height, image_width, _ = image.shape
        image = image.astype("float32")/255.
        image = transform(image=image)["image"][None, ...].to(device)
        pred = model(image)
        pred = pred.argmax(1)[0].numpy().astype("uint8")
        pred = cv2.resize(pred, (image_width, image_height))

        sam_masks = mask_generator.generate(sam_image)
        unified = unified_decision(sam_masks, pred)
        if unified is not None:
          mask_dir = ".".join(img_dirs[i].split("/")[-1].split(".")[:-1]) + ".png"
          cv2.imwrite("./cleaned_data/" + mask_dir, unified)
        

  
