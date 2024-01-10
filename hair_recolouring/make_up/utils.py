import os
import time
from functools import wraps

import cv2
import numpy as np
from alive_progress import alive_bar

NOCOLOR = "\033[0m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
ORANGE = "\033[0;33m"
BLUE = "\033[0;34m"
PURPLE = "\033[0;35m"
CYAN = "\033[0;36m"
LIGHTGRAY = "\033[0;37m"
DARKGRAY = "\033[1;30m"
LIGHTRED = "\033[1;31m"
LIGHTGREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
LIGHTBLUE = "\033[1;34m"
LIGHTPURPLE = "\033[1;35m"
LIGHTCYAN = "\033[1;36m"
WHITE = "\033[1;37m"


# ----------------------------------MEASURE----------------------------------
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(LIGHTBLUE + f"Function {func.__name__} Took {total_time:.5f} seconds")
        return result

    return timeit_wrapper


MEASURE_FOLDER = "measure"


def measure(name_func):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            os.makedirs(MEASURE_FOLDER, exist_ok=True)
            with open(f"{MEASURE_FOLDER}/{name_func}.txt", "a") as f:
                f.write(f"{total_time}\n")
            print(LIGHTBLUE + f"Function {name_func} Took {total_time:.5f} seconds")
            return result

        return wrapper

    return decorator


def measure_calculator():
    func_name = [MEASURE_FOLDER + name for name in os.listdir(MEASURE_FOLDER)]

    for name in func_name:
        try:
            with open(name, "r") as f:
                lines = [eval(line.rstrip()) for line in f]

            print(name + ": ", sum(lines[1:]) / len(lines[1:]))
        except:
            continue


def fps(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = int(1 / (end_time - start_time))
        print(LIGHTBLUE + f"{func.__name__} : {total_time} FPS")
        return result

    return timeit_wrapper


# ----------------------------------INFERENCE----------------------------------


class Inference:
    def __init__(self, predictor) -> None:
        self.predictor = predictor
        self.resolution_dict = {1080: [1920, 1080], 720: [1280, 720]}

    def image(self, image, visualize=False):
        """Process a single image

        Args:
            image (_type_): Input image. Path or np.ndarray
            visualize (bool, optional): Show image output. Default is 'False'
        Returns:
            result (np.ndarray): Output image
        """
        image = cv2.imread(image) if type(image) == "str" else image
        output = self.predictor.run(image)

        if visualize:
            cv2.namedWindow("output", cv2.WINDOW_KEEPRATIO)
            cv2.imshow("output", output)
            cv2.waitKey(0)

        return output

    def webcam(self, name_window="OUTPUT", resolution=720, id_camera=0, length=150):
        """Process with video from webcam

        Args:
            name_window (str, optional): Defaults to 'OUTPUT'.
            resolution (int, optional): Size of window: 720 or 1080. Defaults to 720.
            id_camera (int, optional): Id webcam: 0, 1, 2... Defaults to 0.
            length (int, optional): Length of alive_progress. Defaults to 150.
        """
        print("Using webcam, press [q] to exit, press [s] to save")
        cap = cv2.VideoCapture(id_camera)
        cap.set(3, self.resolution_dict[resolution][0])
        cap.set(4, self.resolution_dict[resolution][1]) * 2

        with alive_bar(theme="musical", length=length) as bar:
            while True:
                _, frame = cap.read()
                frame = cv2.flip(frame, 1)
                frame_origin = frame.copy()

                start = time.time()
                frame = self.predictor.run(frame)
                fps = round(1 / (time.time() - start), 2)

                cv2.putText(
                    frame,
                    "FPS : " + str(fps),
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (50, 170, 50),
                    2,
                )

                frame = cv2.hconcat([frame_origin, frame])
                cv2.imshow(name_window, frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("s"):
                    os.makedirs("results/", exist_ok=True)
                    cv2.imwrite("results/" + str(time.time()) + ".jpg", frame)
                if k == ord("q"):
                    break
                bar()

    def video(self, path_video: str, save_path="results", length=150, concatenation=True, name=None):
        """Process video

        Args:
            name (str): name for video result
            concatenation (bool): Concat input and output if True
            path_video (str): Path video process
            save_path (str, optional): Folder to save video processed. Defaults to 'results'.
            length (int, optional): Length of alive_progress. Defaults to 150.
        """
        if name is None:
            name = os.path.basename(path_video)
        print(f"Processing video {name}\nPlease Uong mieng nuoc & an mieng banh de...")

        cap = cv2.VideoCapture(path_video)
        frame_width = int(cap.get(3)) * 2 if concatenation else int(cap.get(3))
        frame_height = int(cap.get(4))

        size = (frame_width, frame_height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, name)
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc("m", "p", "4", "v"), fps, size)

        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with alive_bar(total=total_frame, theme="musical", length=length) as bar:
            while True:
                _, frame = cap.read()
                if not _:
                    print("Done!")
                    out.release()
                    break
                frame_copy = frame.copy()
                try:
                    result = self.predictor.run(frame)
                    if concatenation:
                        result = cv2.hconcat([frame_copy, result])
                    out.write(result)
                    bar()
                except KeyboardInterrupt:
                    print("Stopped!")
                    out.release()
                    break
        out.release()
        print(f"Video saved in: {save_path}")

    def record_video(self, name="video_record", id_camera=0, fps=15, length=150, resolution=720):
        """Record screen to video

        Args:
            id_camera (int): ID of camera connected with device. Defaults to 0.
            resolution (int): 720 * 1280 or 1080 * 1920
            name (str, optional): Name to save video. Defaults to 'video_record'.
            fps (int, optional): Defaults to 15.
            length (int, optional): Length of alive_progress. Defaults to 150.
        """
        print("Start recoding webcam... \n")
        cap = cv2.VideoCapture(id_camera)
        cap.set(3, self.resolution_dict[resolution][0])
        cap.set(4, self.resolution_dict[resolution][1])
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        os.makedirs("results", exist_ok=True)

        if not name.endswith(".mp4"):
            name = "".join((name, ".mp4"))

        out = cv2.VideoWriter(
            os.path.join("results", name),
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            fps,
            size,
        )

        with alive_bar(theme="musical", length=length) as bar:
            while True:
                _, frame = cap.read()
                frame = cv2.flip(frame, 1)
                out.write(frame)
                bar()
                cv2.imshow("Recoding...", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            out.release()


# ----------------------------------HELPERS----------------------------------


class ExpandBbox:
    def __init__(self, size_expand=1.55):
        self.size_expand = size_expand

    def _check_bbox(self, bbox, img):
        """
        Make sure all coordinates are valid
        """
        h, w = img.shape[:2]
        bbox[0] = 0 if bbox[0] < 0 else bbox[0]
        bbox[1] = 0 if bbox[1] < 0 else bbox[1]
        bbox[2] = w if bbox[2] > w else bbox[2]
        bbox[3] = h if bbox[3] > h else bbox[3]
        return bbox

    def _transform_to_square_bbox(self, bbox, img):
        left, top, right, bottom = bbox[:4]
        old_size = (right - left + bottom - top) / 2
        center_x = right - (right - left) / 2.0
        center_y = bottom - (bottom - top) / 2.0 + old_size * -0.15
        size = int(old_size * self.size_expand)
        roi_box = [0] * 4
        roi_box[0] = center_x - size / 2
        roi_box[1] = center_y - size / 2
        roi_box[2] = roi_box[0] + size
        roi_box[3] = roi_box[1] + size
        roi_box = self._check_bbox(roi_box, img)
        return np.uint32(roi_box)

    def __call__(self, img, bboxes):
        bboxes_sizes = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
        biggest_bbox = bboxes[np.argmax(bboxes_sizes)][:4]
        square_bbox = self._transform_to_square_bbox(biggest_bbox, img)
        return square_bbox


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torch.nn.functional as F


def _diff_x(src, r):
    cum_src = src.cumsum(-2)

    left = cum_src[..., r : 2 * r + 1, :]
    middle = cum_src[..., 2 * r + 1 :, :] - cum_src[..., : -2 * r - 1, :]
    right = cum_src[..., -1:, :] - cum_src[..., -2 * r - 1 : -r - 1, :]

    output = torch.cat([left, middle, right], -2)

    return output


def _diff_y(src, r):
    cum_src = src.cumsum(-1)

    left = cum_src[..., r : 2 * r + 1]
    middle = cum_src[..., 2 * r + 1 :] - cum_src[..., : -2 * r - 1]
    right = cum_src[..., -1:] - cum_src[..., -2 * r - 1 : -r - 1]

    output = torch.cat([left, middle, right], -1)

    return output


def boxfilter2d(src, radius):
    return _diff_y(_diff_x(src, radius), radius)


def guidedfilter2d_color(guide, src, radius, eps, scale=None):
    """guided filter for a color guide image

    Parameters
    -----
    guide: (B, 3, H, W)-dim torch.Tensor
        guide image
    src: (B, C, H, W)-dim torch.Tensor
        filtering image
    radius: int
        filter radius
    eps: float
        regularization coefficient
    """
    assert guide.shape[1] == 3
    if src.ndim == 3:
        src = src[:, None]
    if scale is not None:
        guide_sub = guide.clone()
        src = F.interpolate(src, scale_factor=1.0 / scale, mode="nearest")
        guide = F.interpolate(guide, scale_factor=1.0 / scale, mode="nearest")
        radius = radius // scale

    guide_r, guide_g, guide_b = torch.chunk(guide, 3, 1)  # b x 1 x H x W
    ones = torch.ones_like(guide_r)
    N = boxfilter2d(ones, radius)

    mean_I = boxfilter2d(guide, radius) / N  # b x 3 x H x W
    mean_I_r, mean_I_g, mean_I_b = torch.chunk(mean_I, 3, 1)  # b x 1 x H x W

    mean_p = boxfilter2d(src, radius) / N  # b x C x H x W

    mean_Ip_r = boxfilter2d(guide_r * src, radius) / N  # b x C x H x W
    mean_Ip_g = boxfilter2d(guide_g * src, radius) / N  # b x C x H x W
    mean_Ip_b = boxfilter2d(guide_b * src, radius) / N  # b x C x H x W

    cov_Ip_r = mean_Ip_r - mean_I_r * mean_p  # b x C x H x W
    cov_Ip_g = mean_Ip_g - mean_I_g * mean_p  # b x C x H x W
    cov_Ip_b = mean_Ip_b - mean_I_b * mean_p  # b x C x H x W

    var_I_rr = boxfilter2d(guide_r * guide_r, radius) / N - mean_I_r * mean_I_r + eps  # b x 1 x H x W
    var_I_rg = boxfilter2d(guide_r * guide_g, radius) / N - mean_I_r * mean_I_g  # b x 1 x H x W
    var_I_rb = boxfilter2d(guide_r * guide_b, radius) / N - mean_I_r * mean_I_b  # b x 1 x H x W
    var_I_gg = boxfilter2d(guide_g * guide_g, radius) / N - mean_I_g * mean_I_g + eps  # b x 1 x H x W
    var_I_gb = boxfilter2d(guide_g * guide_b, radius) / N - mean_I_g * mean_I_b  # b x 1 x H x W
    var_I_bb = boxfilter2d(guide_b * guide_b, radius) / N - mean_I_b * mean_I_b + eps  # b x 1 x H x W

    # determinant
    cov_det = (
        var_I_rr * var_I_gg * var_I_bb
        + var_I_rg * var_I_gb * var_I_rb
        + var_I_rb * var_I_rg * var_I_gb
        - var_I_rb * var_I_gg * var_I_rb
        - var_I_rg * var_I_rg * var_I_bb
        - var_I_rr * var_I_gb * var_I_gb
    )  # b x 1 x H x W

    # inverse
    inv_var_I_rr = (var_I_gg * var_I_bb - var_I_gb * var_I_gb) / cov_det  # b x 1 x H x W
    inv_var_I_rg = -(var_I_rg * var_I_bb - var_I_rb * var_I_gb) / cov_det  # b x 1 x H x W
    inv_var_I_rb = (var_I_rg * var_I_gb - var_I_rb * var_I_gg) / cov_det  # b x 1 x H x W
    inv_var_I_gg = (var_I_rr * var_I_bb - var_I_rb * var_I_rb) / cov_det  # b x 1 x H x W
    inv_var_I_gb = -(var_I_rr * var_I_gb - var_I_rb * var_I_rg) / cov_det  # b x 1 x H x W
    inv_var_I_bb = (var_I_rr * var_I_gg - var_I_rg * var_I_rg) / cov_det  # b x 1 x H x W

    inv_sigma = torch.stack(
        [
            torch.stack([inv_var_I_rr, inv_var_I_rg, inv_var_I_rb], 1),
            torch.stack([inv_var_I_rg, inv_var_I_gg, inv_var_I_gb], 1),
            torch.stack([inv_var_I_rb, inv_var_I_gb, inv_var_I_bb], 1),
        ],
        1,
    ).squeeze(
        -3
    )  # b x 3 x 3 x H x W

    cov_Ip = torch.stack([cov_Ip_r, cov_Ip_g, cov_Ip_b], 1)  # b x 3 x C x H x W

    a = torch.einsum("bichw,bijhw->bjchw", (cov_Ip, inv_sigma))
    b = mean_p - a[:, 0] * mean_I_r - a[:, 1] * mean_I_g - a[:, 2] * mean_I_b  # b x C x H x W

    mean_a = torch.stack([boxfilter2d(a[:, i], radius) / N for i in range(3)], 1)
    mean_b = boxfilter2d(b, radius) / N

    if scale is not None:
        guide = guide_sub
        mean_a = torch.stack(
            [F.interpolate(mean_a[:, i], guide.shape[-2:], mode="bilinear") for i in range(3)], 1
        )
        mean_b = F.interpolate(mean_b, guide.shape[-2:], mode="bilinear")

    q = torch.einsum("bichw,bihw->bchw", (mean_a, guide)) + mean_b

    return q


def guidedfilter2d_gray(guide, src, radius, eps, scale=None):
    """guided filter for a gray scale guide image

    Parameters
    -----
    guide: (B, 1, H, W)-dim torch.Tensor
        guide image
    src: (B, C, H, W)-dim torch.Tensor
        filtering image
    radius: int
        filter radius
    eps: float
        regularization coefficient
    """
    if guide.ndim == 3:
        guide = guide[:, None]
    if src.ndim == 3:
        src = src[:, None]

    if scale is not None:
        guide_sub = guide.clone().cuda()
        src = F.interpolate(src, scale_factor=1.0 / scale, mode="nearest")
        guide = F.interpolate(guide, scale_factor=1.0 / scale, mode="nearest")
        radius = radius // scale

    ones = torch.ones_like(guide)
    N = boxfilter2d(ones, radius)

    mean_I = boxfilter2d(guide, radius) / N
    mean_p = boxfilter2d(src, radius) / N
    mean_Ip = boxfilter2d(guide * src, radius) / N
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = boxfilter2d(guide * guide, radius) / N
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = boxfilter2d(a, radius) / N
    mean_b = boxfilter2d(b, radius) / N

    if scale is not None:
        guide = guide_sub
        mean_a = F.interpolate(mean_a, guide.shape[-2:], mode="bilinear")
        mean_b = F.interpolate(mean_b, guide.shape[-2:], mode="bilinear")

    q = mean_a * guide + mean_b
    return q
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


if __name__ == "__main__":
    pass
