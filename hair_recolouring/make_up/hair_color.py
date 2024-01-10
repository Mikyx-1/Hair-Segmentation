import math

import cv2 as cv2
import numpy as np
import torch

from makeup.utils import ExpandBbox
from makeup.utils import guidedfilter2d_color, guidedfilter2d_gray

dict_color_hair = {
    "pupil": {"color": [118, 139, 290], "erode_iteration": 3, "alpha": 0.8},
    "red": {"color": [0, 142, 290], "erode_iteration": 3, "alpha": 0.75},
    "pink": {"color": [159, 92, 290], "erode_iteration": 3, "alpha": 0.75},
    "grey blue": {"color": [108, 106, 290], "erode_iteration": 3, "alpha": 0.75},
    "grey green": {"color": [53, 52, 276], "erode_iteration": 3, "alpha": 0.75},
    "brown": {"color": [9, 114, 276], "erode_iteration": 3, "alpha": 0.82},
    "dark pupil": {"color": [139, 114, 276], "erode_iteration": 3, "alpha": 0.75},
    "dark blue": {"color": [105, 255, 315], "erode_iteration": 3, "alpha": 0.28},
    "light pupil": {"color": [141, 136, 321], "erode_iteration": 3, "alpha": 0.52},
}

new_dict_color_hair = {
    "red": {"color": [0, 0.5725, 1.2275]},
    "orange": {"color": [0.6667, 0.5725, 1.2275]},
    "yellow": {"color": [1.3333, 0.5725, 1.2275]},
    "green": {"color": [2.0, 0.5725, 1.2275]},
    "blue": {"color": [2.6667, 0.5725, 1.2275]},
    "indigo": {"color": [3.3333, 0.5725, 1.2275]},
    "violet": {"color": [4.0, 0.5725, 1.2275]},
}


def make_erode_trimap(mask: np.ndarray, kernel_size=7, iteration=3, alpha=0.1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.float)
    img_erosion = mask.copy()
    for i in range(iteration - 1):
        img_erosion = cv2.erode(img_erosion, kernel=kernel, iterations=1)
        mask += img_erosion
    mask = (mask * alpha) / np.max(mask * alpha)
    return mask


def change_hair_color(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: np.ndarray,
    color_name: str,
    radius=4,
    eps=50,
    threshold=1,
):
    """
    :param image: Image BGR
    :param mask: Mask Grayscale
    :param bbox;
    :param color_name:
    :param radius:
    :param eps:
    :param threshold:
    :return: Image BGR
    """
    colorHSV = dict_color_hair[color_name]["color"]
    alpha = 1.0
    # alpha = dict_color_hair[color_name]['alpha']

    cut_im1 = image[bbox[0] : bbox[2], bbox[1] : bbox[3]]

    cut_im1_hsv = cv2.cvtColor(cut_im1, cv2.COLOR_BGR2HSV).astype(np.float)

    alpha_mask = cv2.ximgproc.guidedFilter(guide=cut_im1, src=mask, radius=radius, eps=eps, dDepth=-1)

    hue_mask = np.where(alpha_mask < threshold / 2, cut_im1_hsv[:, :, 0], colorHSV[0])
    sat_mask = np.where(alpha_mask < threshold / 2, cut_im1_hsv[:, :, 1], colorHSV[1])

    val_alpha = (np.max(cut_im1_hsv[:, :, 2]) - cut_im1_hsv[:, :, 2]) * alpha
    val_mask = np.where(alpha_mask < threshold, cut_im1_hsv[:, :, 2], (colorHSV[2] - val_alpha))

    val_mask = np.where(val_mask < 0, 0, val_mask)
    val_mask = np.where(val_mask > 255, 255, val_mask)

    HSV_mask = cv2.merge(
        (
            hue_mask.astype(np.uint8),
            sat_mask.astype(np.uint8),
            val_mask.astype(np.uint8),
        )
    )

    BGR_mask = cv2.cvtColor(HSV_mask.astype(np.uint8), cv2.COLOR_HSV2BGR)

    mask = cv2.merge((alpha_mask, alpha_mask, alpha_mask)).astype(np.float) / 255.0

    invert_mask = 1.0 - mask

    image_after = cv2.multiply(BGR_mask.astype(np.float), mask) + cv2.multiply(
        cut_im1.astype(np.float), invert_mask
    )

    image[bbox[0] : bbox[2], bbox[1] : bbox[3]] = image_after.astype(np.uint8)
    return image


def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h = (h / 6.0) % 1.0
    h = 2.0 * math.pi * h  # we return 0/2pi output

    return torch.stack((h, s, v), dim=-3)


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    h: torch.Tensor = image[..., 0, :, :] / (2 * math.pi)
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)

    return out


def change_hair_color_ver2(
    image: np.ndarray,
    mask: torch.tensor,
    bbox: torch.tensor,
    color_name: str,
    eps=10,
    alpha=1.0,
    threshold=0,
    rate_guide=115,
    device="cpu",
):
    colorHSV = new_dict_color_hair[color_name]["color"]

    colorHSV = torch.tensor(colorHSV, dtype=torch.float).to(device)
    image_torch = torch.tensor(image, dtype=torch.float).to(device)

    bbox = ExpandBbox(size_expand=1.7)(image, bbox)

    cut_im1 = image_torch[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    mask = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    cut_im1 = cut_im1.flip(-1)
    # RGB image
    cut_im1 = cut_im1.permute((2, 0, 1)).unsqueeze(0)

    cut_im1_hsv = rgb_to_hsv(cut_im1 / 255.0)

    guide = cut_im1
    src = mask.unsqueeze(-1).permute(2, 0, 1)

    radius = min(guide.shape[2], guide.shape[3]) // 10 + 1

    alpha_mask = guidedfilter2d_gray(guide, src, radius, eps, scale=None) - rate_guide
    alpha_mask = torch.clamp(alpha_mask, min=0, max=255)

    hue_mask = torch.where(alpha_mask[:, 0, :, :] < threshold / 2, cut_im1_hsv[:, 0, :, :], colorHSV[0])
    sat_mask = torch.where(alpha_mask[:, 1, :, :] < threshold / 2, cut_im1_hsv[:, 1, :, :], colorHSV[1])
    val_alpha = (torch.max(cut_im1_hsv[:, 2, :, :]) - cut_im1_hsv[:, 2, :, :]) * alpha
    val_mask = torch.where(
        alpha_mask[:, 2, :, :] < threshold,
        cut_im1_hsv[:, 2, :, :],
        (colorHSV[2] - val_alpha),
    )
    val_mask = torch.clamp(val_mask, min=0, max=1)

    HSV_mask = torch.stack((hue_mask, sat_mask, val_mask), 1)

    RGB_mask = hsv_to_rgb(HSV_mask)

    mask = alpha_mask / 255.0
    invert_mask = 1.0 - mask
    image_after = (RGB_mask * mask) + (cut_im1 / 255.0 * invert_mask)

    image[bbox[1] : bbox[3], bbox[0] : bbox[2]] = (
        (image_after * 255).squeeze(0).permute((1, 2, 0)).flip(-1).cpu().detach().numpy().astype(np.uint8)
    )

    return image


def change_hair_color_ver3(
    image: np.ndarray,
    mask: torch.tensor,
    bbox: torch.tensor,
    color_name: str,
    eps=10,
    alpha=1.0,
    threshold=0,
    rate_guide=115,
    return_alpha_mask: bool = False,
):
    colorHSV = new_dict_color_hair[color_name]["color"]

    colorHSV = torch.tensor(colorHSV, dtype=torch.float).cuda()
    image_torch = torch.tensor(image, dtype=torch.float).cuda()

    cut_im1 = image_torch[bbox[0] : bbox[2], bbox[1] : bbox[3], :]
    cut_im1 = cut_im1.flip(-1)
    # RGB image
    cut_im1 = cut_im1.permute((2, 0, 1)).unsqueeze(0)

    cut_im1_hsv = rgb_to_hsv(cut_im1 / 255.0)

    guide = cut_im1
    src = mask.unsqueeze(-1).permute(2, 0, 1)
    radius = min(guide.shape[2], guide.shape[3]) // 10 + 1

    alpha_mask = guidedfilter2d_gray(guide, src, radius, eps, scale=None) - rate_guide
    alpha_mask = torch.clamp(alpha_mask, min=0, max=255)

    hue_mask = torch.where(alpha_mask[:, 0, :, :] < threshold / 2, cut_im1_hsv[:, 0, :, :], colorHSV[0])
    sat_mask = torch.where(alpha_mask[:, 1, :, :] < threshold / 2, cut_im1_hsv[:, 1, :, :], colorHSV[1])
    # val_alpha = (torch.max(cut_im1_hsv[:, 2, :, :]) - cut_im1_hsv[:, 2, :, :]) * alpha
    # val_mask = torch.where(alpha_mask[:, 2, :, :] < threshold, cut_im1_hsv[:, 2, :, :], (colorHSV[2] - val_alpha))
    # val_mask = torch.clamp(val_mask, min=0, max=1)

    val_mask = cut_im1_hsv[:, 2, :, :]

    HSV_mask = torch.stack((hue_mask, sat_mask, val_mask), 1)

    RGB_mask = hsv_to_rgb(HSV_mask)

    mask = alpha_mask / 255.0
    invert_mask = 1.0 - mask
    image_after = (RGB_mask * mask) + (cut_im1 / 255.0 * invert_mask)

    image[bbox[0] : bbox[2], bbox[1] : bbox[3]] = (
        (image_after * 255).squeeze(0).permute((1, 2, 0)).flip(-1).cpu().detach().numpy().astype(np.uint8)
    )

    if not return_alpha_mask:
        return image
    else:
        alpha_mask = alpha_mask.cpu().detach().numpy()
        alpha_mask = alpha_mask.squeeze().transpose(1, 2, 0)
        color_mask = np.zeros_like(image)
        color_mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = alpha_mask
        return image, color_mask


def get_threshold(values_0: torch.Tensor, values_1: torch.Tensor):
    values_0_hist = torch.histc(input=values_0, bins=255, min=0, max=255)
    values_0_hist = values_0_hist / values_0.numel()
    values_0_hist = values_0_hist[1:]
    values_0_hist = values_0_hist / torch.max(values_0_hist)
    values_0_hist_sorted, values_0_hist_indices = torch.sort(values_0_hist)

    values_1_hist = torch.histc(input=values_1, bins=255, min=0, max=255)
    values_1_hist = values_1_hist / values_1.numel()
    values_1_hist = values_1_hist[1:]
    values_1_hist = values_1_hist / torch.max(values_1_hist)
    values_1_hist_sorted, values_1_hist_indices = torch.sort(values_1_hist, descending=True)

    threshold = 0
    for i in range(254):
        if values_0_hist_sorted[i] > values_1_hist_sorted[i]:
            threshold = values_1_hist_indices[i]
            break

    max_values_0_hist = values_0_hist_indices[-1]
    return threshold, max_values_0_hist


def dilatation(src: np.ndarray, shape: int = cv2.MORPH_ELLIPSE, size: int = 20):
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1), (size, size))
    dilatation_dst = cv2.dilate(src, element)
    return dilatation_dst


def change_hair_color_ver6(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: np.ndarray,
    color_name: str,
    eps=10,
    device: torch.device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dilate segmentation mask
    blur_size = 10
    dilated_mask = dilatation(mask, size=blur_size)
    dilated_mask = dilated_mask.astype(np.float32)
    dilated_mask = cv2.GaussianBlur(dilated_mask, (blur_size * 2 + 1, blur_size * 2 + 1), 0)

    # Get brightness mask
    value_channel = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
    value_channel = value_channel[bbox[0] : bbox[2], bbox[1] : bbox[3]]
    value_tensor = torch.tensor(value_channel, dtype=torch.float, device=device)

    hair_value = value_tensor[mask == 255]

    value_hist = torch.histc(hair_value, 255, 0, 255) / hair_value.numel()

    sum = 0
    brightness_threshold = 0
    for i in reversed(list(range(255))):
        sum = sum + value_hist[i]
        if sum > 0.2:
            brightness_threshold = i
            break
    brightness_mask = value_tensor - brightness_threshold
    brightness_mask = torch.clamp(brightness_mask, 0, 255)
    brightness_mask = brightness_mask / 255.0

    # Make tensors
    image_torch = torch.tensor(image, dtype=torch.float, device=device)
    mask = torch.tensor(mask, dtype=torch.float, device=device)
    bbox = torch.tensor(bbox, dtype=torch.int, device=device)
    dilated_mask = torch.tensor(dilated_mask, dtype=torch.float, device=device)

    # Get hsv color
    colorHSV = new_dict_color_hair[color_name]["color"]
    colorHSV = torch.tensor(colorHSV, dtype=torch.float, device=device)

    # Crop image
    cut_im1 = image_torch[bbox[0] : bbox[2], bbox[1] : bbox[3], :]
    cut_im1 = cut_im1.flip(-1)

    # Convert RGB image to HSV
    cut_im1 = cut_im1.permute((2, 0, 1)).unsqueeze(0)
    cut_im1_hsv = rgb_to_hsv(cut_im1 / 255.0)

    # Guide filter alpha mask
    guide = cut_im1
    src = mask.unsqueeze(-1).permute(2, 0, 1)
    radius = min(guide.shape[2], guide.shape[3]) // 10 + 1

    alpha_mask = guidedfilter2d_color(guide, src, radius, eps, scale=None)

    # Normalize alpha mask
    min_alpha_mask = torch.min(alpha_mask)
    max_alpha_mask = torch.max(alpha_mask)
    alpha_mask = (alpha_mask - min_alpha_mask) / (max_alpha_mask - min_alpha_mask) * 255

    # Get threshold by compute histogram
    hair_values = torch.squeeze(alpha_mask)[mask == 255]
    background_values = torch.squeeze(alpha_mask)[mask != 255]

    threshold, most_commond_value = get_threshold(hair_values, background_values)

    threshold = torch.clamp(threshold, 70, 100)

    # Thresholding alpha mask
    alpha_mask = alpha_mask - threshold
    alpha_mask = alpha_mask.clamp(0, most_commond_value - threshold)
    _max = torch.max(alpha_mask)
    alpha_mask = alpha_mask / _max

    # Filter alpha mask by dilated mask
    dilated_mask = dilated_mask / 255.0
    alpha_mask = alpha_mask * dilated_mask

    # Make hsv image
    hue_mask = cut_im1_hsv[:, 0, :, :]
    hue_mask[:, :, :] = colorHSV[0]
    sat_mask = cut_im1_hsv[:, 1, :, :]
    sat_mask[:, :, :] = colorHSV[1]
    val_mask = cut_im1_hsv[:, 2, :, :]

    HSV_image = torch.stack((hue_mask, sat_mask, val_mask), 1)

    # Convert to BGR
    BGR_image = hsv_to_rgb(HSV_image) * 255

    # Blending
    invert_alpha_mask = 1.0 - alpha_mask
    image_after = (BGR_image * alpha_mask) + (cut_im1 * invert_alpha_mask)

    # Enhance brightness
    image_after = image_after * (1 - brightness_mask) + cut_im1 * brightness_mask

    image[bbox[0] : bbox[2], bbox[1] : bbox[3]] = (
        image_after.squeeze(0).permute((1, 2, 0)).flip(-1).cpu().detach().numpy().astype(np.uint8)
    )

    return image
