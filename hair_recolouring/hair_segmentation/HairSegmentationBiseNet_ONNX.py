import cv2
import numpy as np
import onnxruntime


class HairSegmentationBiseNet_ONNX:
    def __init__(self, model_path: str, providers: list = None):
        self.model_path = model_path

        # Model hyper params
        self.input_shape = (360, 480)  # hxw
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Model providers and session
        if providers is None:
            self.providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            self.providers = providers

        self.sess = onnxruntime.InferenceSession(self.model_path, providers=self.providers)
        print(f"[HairSegmentationBiseNet_ONNX] Using providers: {self.sess.get_providers()}")

    def preprocess(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        return image

    def postprocess(self, model_output: np.ndarray):
        model_output = model_output.squeeze(0)
        label = np.argmax(model_output, axis=0)
        label = (1 - label) * 255
        return label

    def run(self, image: np.ndarray):
        inputs = {self.sess.get_inputs()[0].name: self.preprocess(image)}
        model_outputs = self.sess.run(None, inputs)[0]
        label = self.postprocess(model_outputs)
        label = cv2.resize(
            label.astype(np.uint8),
            (
                image.shape[1],
                image.shape[0],
            ),
            cv2.INTER_LINEAR,
        )
        return label


# if __name__ == "__main__":
#     hair_segmentor = HairSegmentationBiseNet_ONNX("/home/os/Downloads/hairsegmentation_bisenet.onnx")
#     image_path = "/home/os/Pictures/extremely-long-layered-hair.jpg"
#     for i in range(100):
#         image = cv2.imread(image_path)
#
#         start_time = time.time()
#         label = hair_segmentor.run(image)
#         print(time.time() - start_time)
#
#     cv2.imshow("image", image)
#     cv2.imshow("label", label)
#     cv2.waitKey(0)
