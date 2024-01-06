import sys
import os
import glob
import random
import time

import numpy as np
from tqdm import tqdm
import pandas as pd

import cv2
import matplotlib.pyplot as plt
import shutil

import torch
from torch import nn, optim
import torch.nn.functional as F

from bisenet import BiSeNet

import argparse
from dataset import createDataLoader, createTestLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description="Example of argparse usage")

parser.add_argument("-d1", "--dirs1", type=str, help="Train Image Folder Directory")
parser.add_argument("-d2", "--dirs2", type=str, help = "Train Mask Folder Directory")
parser.add_argument("-d3", "--dirs3", type=str, help = "Validation Image Folder Directory")
parser.add_argument("-d4", "--dirs4", type=str, help = "Validation Mask Folder Directory")
parser.add_argument("-d5", "--dirs5", type=str, help="Test Image Folder Directory")
parser.add_argument("-d6", "--dirs6", type=str, help="Test Mask Folder Directory")


args = parser.parse_args()

train_img_dirs = sorted(glob.glob(args.dirs1 + "/*"))
train_mask_dirs = sorted(glob.glob(args.dirs2 + "/*"))

val_img_dirs = sorted(glob.glob(args.dirs3 + "/*"))
val_mask_dirs = sorted(glob.glob(args.dirs4 + "/*"))

test_img_dirs = sorted(glob.glob(args.dirs5 + "/*"))
test_mask_dirs = sorted(glob.glob(args.dirs6 + "/*"))



print(f"Num train img dirs: {len(train_img_dirs)}, Num train img masks: {len(train_mask_dirs)}")
print(f"Num val img dirs: {len(val_img_dirs)}, Num val img masks: {len(val_mask_dirs)}")

train_loader = createDataLoader(train_img_dirs, train_mask_dirs, val_set=False, shuffle=True, batch_size=32, num_workers=4)
val_loader = createDataLoader(val_img_dirs, val_mask_dirs,val_set = True, shuffle=False, batch_size=32, num_workers=4)
test_loader = createTestLoader(test_img_dirs, test_mask_dirs, batch_size=32, num_workers=4)



model = BiSeNet(2)
model.to(device)
auxiliary_loss_coeff = 1
num_epochs = 100
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3, amsgrad=True)


def train_1_epoch():
    model.train()
    model.activate_training_mode()
    loss_value = 0
    cnt = 0
    for (x_train, y_train) in tqdm(train_loader):
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        logits, aux_logits1, aux_logits2 = model(x_train)
        train_loss = loss_fn(logits, y_train) + auxiliary_loss_coeff*loss_fn(aux_logits1, y_train) + auxiliary_loss_coeff*loss_fn(aux_logits2, y_train)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        loss_value += train_loss.item()
        cnt += 1
    return loss_value/cnt


@torch.no_grad()
def calculateIOU_Segmentation(pred: torch.Tensor, label: torch.Tensor):
  pred = pred.argmax(1)
  intersection = 2*(pred*label).sum(dim=(1, 2))
  intersection = torch.where(intersection==0, 0.01, intersection)
  union = label.sum(dim=(1, 2)) + pred.sum(dim=(1, 2))
  union = torch.where(union==0, 0.01, union)
  return (intersection/union).mean()

@torch.no_grad()
def evaluate():
   model.eval()
   model.activate_evaluation_mode()
   loss_value = 0
   cnt = 0
   sumIoU = 0.
   for x_val, y_val in val_loader:
      x_val = x_val.to(device)
      y_val = y_val.to(device)
      logits = model(x_val)
      val_loss = loss_fn(logits, y_val)
      sumIoU += calculateIOU_Segmentation(logits, y_val).cpu().item()
      loss_value += val_loss.item()
      cnt += 1

   return loss_value/cnt, sumIoU/cnt

@torch.no_grad()
def evaluate_test_set():
   model.eval()
   model.activate_evaluation_mode()
   sumIoU = 0.
   cnt = 0
   for x_test, y_test in test_loader:
      x_test = x_test.to(device)
      y_test = y_test.to(device)
      logits = model(x_test)
      sumIoU += calculateIOU_Segmentation(logits, y_test).cpu().item()
      cnt += 1
   return sumIoU/cnt


train_losses = []
val_losses = []
val_IOU = []
test_IOU_hist = []

def train(epochs):
   min_train_loss = 1
   min_val_loss = 1
   val_maxIOU = 0
   test_maxIOU = 0
   for epoch in range(1, epochs+1):
    train_loss = train_1_epoch()
    val_loss, val_meanIOU = evaluate()
    testIoU = evaluate_test_set()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_IOU.append(val_meanIOU)
    test_IOU_hist.append(testIoU)

    if train_loss < min_train_loss:
       min_train_loss = train_loss
       torch.save(model.state_dict(), "train_loss_min.pt")
    
    if val_loss < min_val_loss:
       min_val_loss = val_loss
       torch.save(model.state_dict(), "val_loss_min.pt")

    if val_meanIOU > val_maxIOU:
       val_maxIOU = val_meanIOU
       torch.save(model.state_dict(), "val_maxIOU.pt")

    if testIoU > test_maxIOU:
      test_maxIOU = testIoU
      torch.save(model.state_dict(), "test_maxIOU.pt")
        
    torch.save(model.state_dict(), "last.pt")
   
    train_hist = pd.DataFrame({"Epoch": [i for i in range(1, epoch+1)], "train_loss": train_losses, "val_loss": val_losses, "val_IOU": val_IOU, "test_IOU": test_IOU_hist})
    train_hist.to_csv("./train_history.csv")
    print(f"Epoch: {epoch}/{epochs} Train Loss: {train_loss} Val Loss: {val_loss} Val MeanIoU: {val_meanIOU}, Test MeanIOU: {testIoU}")
   return train_losses, val_losses, val_IOU

train_losses, val_losses, val_IOU = train(num_epochs)
# train_hist = pd.DataFrame({"Epoch": [i for i in range(1, num_epochs+1)], "train_loss": train_losses, "val_loss": val_losses, "val_IOU": val_IOU})
# train_hist.to_csv("./train_history.csv")
