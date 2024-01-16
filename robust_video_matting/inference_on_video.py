import av
import os
import pims
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import time

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VideoReader(Dataset):
    def __init__(self, path, transform=None):
        self.video = pims.PyAVVideoReader(path)
        self.rate = self.video.frame_rate
        self.transform = transform
        
    @property
    def frame_rate(self):
        return self.rate
    
    def __len__(self):
        return len(self.video)
    
    def __getitem__(self, idx):
        frame = self.video[idx]
        frame = Image.fromarray(np.asarray(frame))
        if self.transform is not None:
            frame = self.transform(frame)
        return frame

def auto_downsample_ratio(h, w):
    return min(512/max(h, w), 1)

transform = transforms.Compose([
    transforms.Resize((360, 360)), 
    transforms.ToTensor()
])

source = VideoReader("./3_0.mp4", transform)
model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")
model = model.eval()
model = model.to(device)
param = next(model.parameters())
dtype = param.dtype

downsample_ratio = 1
rec = [None]*4

for i in range(len(source)):
    src = source[i]
    fgr, pha, *rec = model(src.unsqueeze(0), *rec, downsample_ratio)
    break
    
