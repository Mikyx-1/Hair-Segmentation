'''
Implementation of data cleaning with available labels
'''


#!pip install opencv-python pycocotools matplotlib onnxruntime onnx


# Run successfully
#!pip install git+https://github.com/facebookresearch/segment-anything.git

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

img_dirs = sorted(glob.glob("/kaggle/input/dirty-data-1/JPEGImages/*"))
mask_dirs = sorted(glob.glob("/kaggle/input/dirty-data-1/SegmentationClassAug/*"))
print(f"Num images: {len(img_dirs)}, Num masks: {len(mask_dirs)}")

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["default"](checkpoint="/kaggle/input/sam-ckpt/sam_vit_h_4b8939.pth")
sam.to(device="cuda")
mask_generator = SamAutomaticMaskGenerator(sam)

for img_idx in tqdm(range(len(img_dirs))):
    image = cv2.imread(img_dirs[img_idx])[:, :, ::-1]
    gt = np.where(cv2.imread(mask_dirs[img_idx], cv2.IMREAD_GRAYSCALE)==2, 1, 0)
    masks = mask_generator.generate(image)
    
    max_iou = 0
    most_likely_mask = None
    second_likely_mask = None
    for i in range(len(masks)):
        mask = masks[i]["segmentation"].astype("uint8")
        iou = 2*(mask*gt).sum()/(mask.sum() + gt.sum())
        if iou > max_iou and iou > 0.85:
            max_iou = iou
            most_likely_mask = mask
            
        if iou > max_iou and iou > 0.75:
            max_iou = iou
            second_likely_mask = mask
            
    if most_likely_mask is not None:
        cv2.imwrite("cleaned_data_1_0.85/" + mask_dirs[img_idx].split("/")[-1], most_likely_mask)
    if most_likely_mask is None and second_likely_mask is not None:
        cv2.imwrite("cleaned_data_1_0.75//" + mask_dirs[img_idx].split("/")[-1], second_likely_mask)

  
