import time
import tensorflow as tf
import glob
import cv2
import numpy as np
import random
import onnxruntime as ort
import matplotlib.pyplot as plt

def speedtest_mediapipe_on_videos(model_path, video_dirs):
    global_fps = 0.
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        model = ort.InferenceSession(model_path, providers = ["CUDAExecutionProvider"])
    else:
        model = ort.InferenceSession(model_path, providers = ["CPUExecutionProvider"])
    
    for vid_dir in video_dirs:
        sum_fps = 0.
        cnt = 0
        cap = cv2.VideoCapture(vid_dir)
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (256, 256)).astype(np.float32)/255.0
                image = image[None , ...]
                
                pred = model.run([], {"input_29": image})[0]
                pred = pred[0].argmax(-1)
                sum_fps += 1/(time.time() - start_time)
                cnt += 1
            else:
                break
        global_fps += sum_fps/cnt
        
    return global_fps/len(video_dirs)
    

# 13.438s for 100 images
def speedtest_prev_bisenet_videos(model_path, video_dirs):
    global_fps = 0.
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        model = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
    else:
        model = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
    model = ort.InferenceSession(model_path, providers=ort.get_available_providers())
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_name = model.get_inputs()[0].name
    
    for vid_dir in video_dirs:
        cnt = 0
        sum_fps = 0.
        cap = cv2.VideoCapture(vid_dir)
        
        while cap.isOpened():
            start_time = time.time()
            ret, image = cap.read()
            if ret:              
                image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (360, 480))
                image = image.astype(np.float32)/255.0
                image = (image - mean)/std
                image = np.transpose(image, (2, 0, 1))
                image = np.expand_dims(image, 0)
                
                pred = model.run([], {input_name: image})[0]
                pred = pred.squeeze(0)
                pred = np.argmax(pred, 0)
                sum_fps += 1/(time.time() - start_time)
                cnt += 1
            else:
                break
        global_fps += sum_fps/cnt
        
    return global_fps/len(video_dirs)

def speedtest_new_bisenet_videos(model_path, video_dirs):
    global_fps = 0.
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        model = ort.InferenceSession(model_path, providers = ["CUDAExecutionProvider"])
    else:
        model = ort.InferenceSession(model_path, providers = ["CPUExecutionProvider"])
    
    for vid_dir in video_dirs:
        sum_fps = 0.
        cnt = 0
        cap = cv2.VideoCapture(vid_dir)
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (360, 360)).astype(np.float32)/255.0
                image = image.transpose((2, 0, 1))[None , ...]
                
                pred = model.run([], {"input": image})[0]
                pred = pred[0].argmax(0)
                sum_fps += 1/(time.time() - start_time)
                cnt += 1
            else:
                break
        global_fps += sum_fps/cnt
    return global_fps/len(video_dirs)


def speedtest_deeplabv3plus_videos(model_path, video_dirs):
  '''
  Speed test for deeplabv3+ on videos
  '''
    global_fps = 0.
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        model = ort.InferenceSession(model_path, providers = ["CUDAExecutionProvider"])
    else:
        model = ort.InferenceSession(model_path, providers = ["CPUExecutionProvider"])
    for vid_dir in video_dirs:
        sum_fps = 0.
        cnt = 0
        cap = cv2.VideoCapture(vid_dir)
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (256, 256)).astype(np.float32)/255.0 # Image size for deeplabV3+
                image = image.transpose((2, 0, 1))[None , ...]
                
                pred = model.run([], {"input": image})[0]
                pred = pred[0].argmax(0)
                sum_fps += 1/(time.time() - start_time)
                cnt += 1
            else:
                break
        
        global_fps += sum_fps/cnt
    return global_fps/len(video_dirs)
