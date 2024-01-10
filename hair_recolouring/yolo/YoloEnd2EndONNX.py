import sys
from typing import List

import cv2
import numpy as np
import onnxruntime

sys.path.insert(0, ".")
from yolo.general import scale_coords
from yolo.plots import Annotator, colors
from yolo.utils import check_img_size, letterbox


class YoloEnd2EndONNX:
    __COCO_CLASS_NAMES = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(
        self,
        model_path: str,
        providers: List[tuple] = None,
        configs: dict = None,
    ):
        if providers is None:
            self.providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            self.providers = providers

        self.session = onnxruntime.InferenceSession(model_path, providers=self.providers)
        meta = self.session.get_modelmeta().custom_metadata_map  # metadata
        self.stride, self.names = 32, self.__COCO_CLASS_NAMES  # assign defaults
        if "stride" in meta:
            self.stride, self.names = int(meta["stride"]), eval(meta["names"])
        self.input_shape = configs["input_shape"]
        self.input_shape = check_img_size(self.input_shape, s=self.stride)

    def visualize(
        self,
        image: np.ndarray,
        predict_result: np.ndarray,
        line_thickness: int = 3,
    ):
        """
        Visualize detection results
        :param image:
        :param predict_result:
        :param line_thickness: Bounding box line thickness
        :return:
        """
        annotator = Annotator(image, line_width=line_thickness, example=str(self.names))
        for pred in predict_result:
            if len(pred):
                for *xyxy, cls, conf in reversed(pred[:, 1:]):
                    c = int(cls)  # integer class
                    label = f"{self.names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))
            return annotator.result()

    def __call__(self, img_bgr: np.ndarray):
        """

        :param img_bgr:
        :return: [batch_size, num_bboxes, 7]. The third dimensions is in format [num_bboxes, x0, y0, x1, y1, class_id,
        confidence_score] where x0, y0, x1, y1 are in input image coordinates
        """

        # Preprocess
        preprocessed_frame, ratio, (dw, dh) = letterbox(
            img_bgr, new_shape=self.input_shape, stride=self.stride, auto=False
        )
        preprocessed_frame = np.stack([preprocessed_frame], 0)

        # Convert
        preprocessed_frame = preprocessed_frame[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        preprocessed_frame = np.ascontiguousarray(preprocessed_frame)

        preprocessed_frame = preprocessed_frame.astype(np.float32)
        preprocessed_frame /= 255  # 0 - 255 to 0.0 - 1.0
        if len(preprocessed_frame.shape) == 3:
            preprocessed_frame = preprocessed_frame[None]  # expand for batch dim

        # Run inference
        y = self.session.run(
            [self.session.get_outputs()[0].name],
            {self.session.get_inputs()[0].name: preprocessed_frame},
        )

        for i in range(len(y)):
            if len(y[i]):
                # Rescale boxes from img_size to im0 size
                y[i][:, 1:5] = scale_coords(self.input_shape, y[i][:, 1:5], img_bgr.shape).round()
        return y


# Below is model inference test code
if __name__ == "__main__":
    yoloEnd2EndONNX = YoloEnd2EndONNX(
        model_path="saved_weights/yolo/yolov6n_end2end.onnx",
        providers=[
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ],
        configs={"input_shape": [320, 320]},
    )

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # Run inference
        output = yoloEnd2EndONNX(frame)

        # Visualize
        display_image = yoloEnd2EndONNX.visualize(frame, output)
        cv2.imshow("display_image", display_image)
        if cv2.waitKey(1) == ord("q"):
            break
    capture.release()
