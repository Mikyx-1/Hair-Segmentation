import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
from hair_segmentation.HairSegmentationBiseNet_ONNX import (
    HairSegmentationBiseNet_ONNX,
)
from yolo.YoloEnd2EndONNX import YoloEnd2EndONNX
from makeup.hair_color import change_hair_color_ver2, change_hair_color_ver6


def run_change_hair_color_v2(frame: np.ndarray, mask: np.ndarray, color_name: str) -> np.ndarray:
    """
    Run changing hair color v2
    @param frame: BGR image frame
    @param mask: segmentation mask: 255 - hair, 0 - other
    @param color_name: Preset color name: ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    @return:
    """

    # Get bbox
    y, x = np.where(mask != 0)
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    bbox_padding_rate = 1.5

    center_x = 0.5 * (x_max + x_min)
    center_y = 0.5 * (y_max + y_min)
    bbox_h = bbox_padding_rate * (y_max - y_min)
    bbox_w = bbox_padding_rate * (x_max - x_min)
    x_0 = max(0, int(center_x - bbox_w / 2))
    y_0 = max(0, int(center_y - bbox_h / 2))
    x_1 = min(frame.shape[1], int(center_x + bbox_w / 2))
    y_1 = min(frame.shape[0], int(center_y + bbox_h / 2))

    # Perform hair color changing
    cropped_mask = mask[y_0:y_1, x_0:x_1]

    output_image_v2 = change_hair_color_ver2(
        frame.copy(),
        torch.tensor(cropped_mask).cuda(),
        bbox=torch.tensor([y_0, x_0, y_1, x_1]).cuda(),
        color_name=color_name,
    )
    return output_image_v2


def run_change_hair_color_v6(
    frame: np.ndarray, mask: np.ndarray, color_name: str, device: torch.device
) -> np.ndarray:
    """
    Run changing hair color v6
    @param frame: BGR image frame
    @param mask: segmentation mask: 255 - hair, 0 - other
    @param color_name: Preset color name: ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    @param device: Torch device
    @return:
    """

    # Get bbox
    y, x = np.where(mask != 0)
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    bbox_padding_rate = 1.5

    center_x = 0.5 * (x_max + x_min)
    center_y = 0.5 * (y_max + y_min)
    bbox_h = bbox_padding_rate * (y_max - y_min)
    bbox_w = bbox_padding_rate * (x_max - x_min)
    x_0 = max(0, int(center_x - bbox_w / 2))
    y_0 = max(0, int(center_y - bbox_h / 2))
    x_1 = min(frame.shape[1], int(center_x + bbox_w / 2))
    y_1 = min(frame.shape[0], int(center_y + bbox_h / 2))

    # Perform hair color changing
    cropped_mask = mask[y_0:y_1, x_0:x_1]

    output_image_v6 = change_hair_color_ver6(
        image=frame,
        mask=cropped_mask,
        bbox=np.array([y_0, x_0, y_1, x_1]),
        color_name=color_name,
        device=device,
    )
    return output_image_v6


def test_changing_hair_color(
    video_path: str,
    version: int,
    color_name: str,
    hair_segmentation_model_path: str = "./hairsegmentation_bisenet.onnx",
    yolo_model_path: str = "./yolov6n_end2end.onnx",
    yolo_configs: dict = None,
):
    """

    @param video_path: Path to video source
    @param version: Change hair color version: [2, 6]
    @param color_name: Preset color name: ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    @param hair_segmentation_model_path: Path to hair segmentation model
    @param yolo_model_path: Path to yolo model
    @param yolo_configs: Yolo configs
    @return:
    """
    if yolo_configs is None:
        yolo_configs = {"input_shape": [320, 320]}

    assert version in [2, 6]

    # Init models
    hair_segmentor = HairSegmentationBiseNet_ONNX(model_path=hair_segmentation_model_path)
    human_detector = YoloEnd2EndONNX(model_path=yolo_model_path, configs=yolo_configs)

    capture = cv2.VideoCapture(video_path)

    # Runtime
    human_detection_time = 0
    human_detection_count = 0

    hair_segmentation_time = 0
    hair_segmentation_count = 0

    change_hair_color_time = 0
    change_hair_color_count = 0

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        cv2.imshow("original_frame", frame)

        # Run human detection
        start_time = time.time()
        detection_output = human_detector(frame)
        human_detection_time += time.time() - start_time
        human_detection_count += 1

        # Get the largest human bounding box
        largest_bbox = None
        largest_bbox_size = None

        for detection in detection_output[0]:  # num_bboxes, x1, y1, x2, y2, class_id, confidence
            # Human class
            if detection[5] == 0:
                if largest_bbox is None:
                    largest_bbox = detection[1:5]
                    largest_bbox_size = (largest_bbox[2] - largest_bbox[0]) * (
                        largest_bbox[3] - largest_bbox[1]
                    )
                else:
                    bbox_size = (detection[3] - detection[1]) * (detection[4] - detection[2])
                    if bbox_size > largest_bbox_size:
                        largest_bbox_size = bbox_size
                        largest_bbox = detection[1:5]

        if largest_bbox is not None:
            largest_bbox = np.floor(largest_bbox).astype(np.int)
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            # Run hair segmentation
            start_time = time.time()
            bbox_mask = hair_segmentor.run(
                frame[largest_bbox[1] : largest_bbox[3], largest_bbox[0] : largest_bbox[2]]
            )

            hair_segmentation_time += time.time() - start_time
            hair_segmentation_count += 1

            mask[largest_bbox[1] : largest_bbox[3], largest_bbox[0] : largest_bbox[2]] = bbox_mask

            # Run change hair color

            start_time = time.time()
            if version == 6:
                output = run_change_hair_color_v6(frame, mask, color_name, torch.device("cuda"))
            else:
                output = run_change_hair_color_v2(frame, mask, color_name)
            change_hair_color_time += time.time() - start_time
            change_hair_color_count += 1

            cv2.imshow(f"change_hair_v{version}", output)

        print(" ------------------------ ")
        print(f"AVG Human detection time    :\t {human_detection_time/human_detection_count} s")
        print(f"AVG Hair segmentation time  :\t {hair_segmentation_time/hair_segmentation_count} s")
        print(f"AVG Changing hair color time:\t {change_hair_color_time/change_hair_color_count} s")

        if cv2.waitKey(1) == ord("q"):
            break

    capture.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test change hair color using hair segmentation model")
    parser.add_argument("--video_path", type=str, default=None, help="Path to input video")
    parser.add_argument("--version", type=int, choices=[2, 6], help="Changing hair color version")
    parser.add_argument(
        "--color",
        choices=["red", "orange", "yellow", "green", "blue", "indigo", "violet"],
        default="red",
        help="New hair color name",
    )
    args = parser.parse_args()
    test_changing_hair_color(video_path=args.video_path, version=args.version, color_name=args.color)
