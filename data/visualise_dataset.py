import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import os
import random

img_dirs = sorted(glob.glob("./evaluation_set/images/*"))
mask_dirs = sorted(glob.glob("./evaluation_set/masks/*"))
print(f"Num images: {len(img_dirs)}, Num masks: {len(mask_dirs)}")


for idx in range(64):
    image = cv2.cvtColor(cv2.imread(img_dirs[idx]), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_dirs[idx], cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask==2, 255, 0)

    mask = np.stack([np.zeros_like(mask), mask, np.zeros_like(mask)], -1).astype("uint8")

    alpha = 0.5
    output = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 12))
    ax1.imshow(image)
    ax2.imshow(output)
