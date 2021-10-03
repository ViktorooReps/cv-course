from typing import Tuple

import numpy as np


def decompose(img: np.array) -> Tuple[np.array, np.array, np.array]:
    return img[:, :, 0], img[:, :, 1], img[:, :, 2]


def compose(red: np.array, green: np.array, blue: np.array) -> np.array:
    return np.dstack([red, green, blue])


def convert_to_brightness_map(img: np.array) -> np.array:
    red, green, blue = decompose(img)
    return red * 0.299 + green * 0.587 + blue * 0.114


def compute_energy(img: np.array):
    brightness = convert_to_brightness_map(img)

    x_shifted = np.pad(brightness, [(0, 2), (0, 0)], mode='edge').astype(float)
    orig_x_expanded = np.pad(brightness, [(2, 0), (0, 0)], mode='edge').astype(float)

    y_shifted = np.pad(brightness, [(0, 0), (0, 2)], mode='edge').astype(float)
    orig_y_expanded = np.pad(brightness, [(0, 0), (2, 0)], mode='edge').astype(float)

    x_grad = (orig_x_expanded - x_shifted)[1:-1, :]
    y_grad = (orig_y_expanded - y_shifted)[:, 1:-1]

    return np.sqrt(x_grad ** 2 + y_grad ** 2)


def compute_seam_energy(prev_seam_eng: np.array, curr_img_eng: np.array):
    shifted_stack = np.dstack([
        np.pad(prev_seam_eng, (2, 0), mode='maximum'),
        np.pad(prev_seam_eng, (1, 1), mode='maximum'),
        np.pad(prev_seam_eng, (0, 2), mode='maximum')
    ])

    minimums = np.min(shifted_stack, axis=-1)[1:-1]
    return curr_img_eng + minimums


def compute_seam_mask(brightness_map: np.array, axis: int = 0):
    pass


def remove_seam(img: np.array, brightness_map: np.array, axis: int = 0):
    pass


def seam_carve():
    pass
