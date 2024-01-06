import time
import tensorflow as tf
import glob
import cv2
import numpy as np
import random
import onnxruntime as ort
import matplotlib.pyplot as plt
import pandas as pd
from utils import speedtest_mediapipe_on_videos, \
    speedtest_prev_bisenet_videos, speedtest_deeplabv3plus_videos, \
    speedtest_new_bisenet_videos




video_dirs = glob.glob("./videos/*")
mediapipe_model_path = "selfie_multiclass_256x256.onnx"
prev_bisenet_path = "hairsegmentation_bisenet.onnx"
deeplabv3plus_path = "deeplabv3plus.onnx"
new_bisenet_path = "bisenet.onnx"


#                                       Old BiSe
#########################################################################################
mean_fps_old_bise = speedtest_prev_bisenet_videos(prev_bisenet_path, video_dirs)
df = pd.DataFrame({"Model": ["BiseNet Old"], 
                   "Mean FPS: ": [mean_fps_old_bise]})
df.to_csv("FPS_test.csv", index=None)
#########################################################################################

#                                  Mediapipe Model
#########################################################################################
mean_fps_mediapipe = speedtest_mediapipe_on_videos(mediapipe_model_path, video_dirs)

df = pd.DataFrame({"Model": ["BiseNet Old", "Mediapipe Model"], 
                   "Mean FPS: ": [mean_fps_old_bise, mean_fps_mediapipe]})
df.to_csv("FPS_test.csv", index=None)
##########################################################################################

#                                       DeepLabV3+
###################################################################################################
mean_fps_deeplabv3plus = speedtest_deeplabv3plus_videos(deeplabv3plus_path, video_dirs)
df = pd.DataFrame({"Model": ["BiseNet Old", "Mediapipe Model", "DeepLabV3+"], 
                   "Mean FPS: ": [mean_fps_old_bise, mean_fps_mediapipe, mean_fps_deeplabv3plus]})
df.to_csv("FPS_test.csv", index=None)
###################################################################################################

#                                         # BiseNet New
#######################################################################################################################
mean_fps_new_bise = speedtest_new_bisenet_videos(new_bisenet_path, video_dirs)

df = pd.DataFrame({"Model": ["BiseNet Old", "Mediapipe Model", "DeepLabV3+", "New BiSeNet"], 
                   "Mean FPS: ": [mean_fps_old_bise, mean_fps_mediapipe, mean_fps_deeplabv3plus, mean_fps_new_bise]})

df.to_csv("FPS_test.csv", index=None)
########################################################################################################################
