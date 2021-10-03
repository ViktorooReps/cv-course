import enum
from functools import partial
from pprint import pprint
from typing import Tuple, List, Optional

import numpy as np
from numpy.typing import NDArray


def decompose(img: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    return img[:, :, 0].astype(np.float64), img[:, :, 1].astype(np.float64), img[:, :, 2].astype(np.float64)


def compose(red: NDArray, green: np.array, blue: np.array) -> NDArray:
    return np.dstack([red, green, blue]).astype(np.int8)


def convert_to_brightness_map(img: NDArray) -> NDArray:
    red, green, blue = decompose(img)
    return red * 0.299 + green * 0.587 + blue * 0.114


def compute_energy(img: NDArray):
    brightness = convert_to_brightness_map(img)

    x_shifted = np.pad(brightness, [(0, 2), (0, 0)], mode='edge').astype(np.float64)
    orig_x_expanded = np.pad(brightness, [(2, 0), (0, 0)], mode='edge').astype(np.float64)

    y_shifted = np.pad(brightness, [(0, 0), (0, 2)], mode='edge').astype(np.float64)
    orig_y_expanded = np.pad(brightness, [(0, 0), (2, 0)], mode='edge').astype(np.float64)

    x_grad = (orig_x_expanded - x_shifted)[1:-1, :]
    y_grad = (orig_y_expanded - y_shifted)[:, 1:-1]

    return np.sqrt(x_grad ** 2 + y_grad ** 2)


def compute_seam_energy(prev_seam_eng: NDArray, curr_img_eng: NDArray) -> NDArray:
    shifted_stack = np.stack([
        np.pad(prev_seam_eng, (2, 0), mode='maximum'),
        np.pad(prev_seam_eng, (1, 1), mode='maximum'),
        np.pad(prev_seam_eng, (0, 2), mode='maximum')
    ]).astype(np.float64)

    minimums = np.min(shifted_stack, axis=0)[1:-1].astype(np.float64)
    return curr_img_eng + minimums


def compute_seam_mask(energy_map: NDArray, axis: int = 0) -> NDArray:
    if axis == 0:
        seam_energy: List[np.array] = [np.ravel(energy_map[0])]
        for i in range(1, energy_map.shape[0]):
            curr = compute_seam_energy(seam_energy[-1], np.ravel(energy_map[i]))
            seam_energy.append(curr)

        def choose_span(idx: int, pidx: Optional[int]):
            return ((idx >= pidx - 1) and (idx <= pidx + 1)) if pidx is not None else True

        indexes: List[int] = []
        prev_idx = None
        for seam_row_energy in reversed(seam_energy):
            prev_idx = next(filter(partial(choose_span, pidx=prev_idx), np.argsort(seam_row_energy, kind='stable')))
            indexes.append(prev_idx)
        indexes.reverse()

        def one_hot(_idx: int) -> np.array:
            res = np.zeros(energy_map.shape[1]).astype(bool)
            res[_idx] = True
            return res

        return np.array([one_hot(idx) for idx in indexes], dtype=bool)
    elif axis == 1:
        seam_energy: List[np.array] = [np.ravel(energy_map[:, 0])]
        for i in range(1, energy_map.shape[1]):
            curr = compute_seam_energy(seam_energy[-1], np.ravel(energy_map[:, i]))
            seam_energy.append(curr)

        def choose_span(idx: int, pidx: Optional[int]):
            return ((idx >= pidx - 1) and (idx <= pidx + 1)) if pidx is not None else True

        indexes: List[int] = []
        prev_idx = None
        for seam_row_energy in reversed(seam_energy):
            prev_idx = next(filter(partial(choose_span, pidx=prev_idx), np.argsort(seam_row_energy, kind='stable')))
            indexes.append(prev_idx)
        indexes.reverse()

        def one_hot(_idx: int) -> np.array:
            res = np.zeros(energy_map.shape[0]).astype(bool)
            res[_idx] = True
            return res

        return np.array([one_hot(idx) for idx in indexes], dtype=bool).T
    else:
        raise ValueError


def remove_seam(img: NDArray, mask: NDArray, energy_map: NDArray, axis: int = 0) -> Tuple[NDArray, NDArray, NDArray]:
    new_shape = list(img.shape)[:2]
    new_shape[1 - axis] -= 1

    red, green, blue = decompose(img)
    seam_mask = compute_seam_mask(energy_map, axis=axis)

    return compose(red[~seam_mask].reshape(new_shape),
                   green[~seam_mask].reshape(new_shape),
                   blue[~seam_mask].reshape(new_shape)), mask[~seam_mask].reshape(new_shape), seam_mask


def expand_seam(img: NDArray, mask: NDArray, energy_map: NDArray, axis: int = 0) -> Tuple[NDArray, NDArray, NDArray]:
    new_shape = list(img.shape)[:2]
    new_shape[1 - axis] += 1

    pad = [(1, 0), (0, 0)] if axis else [(0, 0), (1, 0)]
    rev_pad = [(0, 1), (0, 0)] if axis else [(0, 0), (0, 1)]

    red, green, blue = decompose(img)
    unpadded_mask = compute_seam_mask(energy_map, axis=axis)
    seam_mask = np.pad(unpadded_mask, pad)
    copy_mask = np.pad(unpadded_mask, rev_pad)

    new_red = np.zeros(new_shape)
    new_green = np.zeros(new_shape)
    new_blue = np.zeros(new_shape)
    new_mask = np.zeros(new_shape)

    orig_red_seam = red[unpadded_mask]
    orig_green_seam = green[unpadded_mask]
    orig_blue_seam = blue[unpadded_mask]

    new_red[~seam_mask] = red.ravel()
    new_red[seam_mask] = (new_red[copy_mask] + orig_red_seam) // 2

    new_green[~seam_mask] = green.ravel()
    new_green[seam_mask] = (new_green[copy_mask] + orig_green_seam) // 2

    new_blue[~seam_mask] = blue.ravel()
    new_blue[seam_mask] = (new_blue[copy_mask] + orig_blue_seam) // 2

    new_mask[~seam_mask] = mask.ravel()
    new_mask[seam_mask] = 1

    return compose(new_red, new_green, new_blue), new_mask, unpadded_mask


class Mode(enum.Enum):
    HORIZONTAL_SHRINK = 'horizontal shrink'
    VERTICAL_SHRINK = 'vertical shrink'
    HORIZONTAL_EXPAND = 'horizontal expand'
    VERTICAL_EXPAND = 'vertical expand'


def seam_carve(img: NDArray, mode: str, mask: Optional[NDArray] = None) -> Tuple[NDArray, NDArray, NDArray]:
    mode = Mode(mode)
    if mask is None:
        mask = np.zeros([img.shape[0], img.shape[1]]).astype(np.float64)
    energy_map = compute_energy(img) + mask * 256 * 100
    if mode == Mode.VERTICAL_EXPAND:
        carved_img, carved_mask, seam_mask = expand_seam(img, mask, energy_map, axis=1)
    elif mode == Mode.HORIZONTAL_EXPAND:
        carved_img, carved_mask, seam_mask = expand_seam(img, mask, energy_map, axis=0)
    elif mode == Mode.VERTICAL_SHRINK:
        carved_img, carved_mask, seam_mask = remove_seam(img, mask, energy_map, axis=1)
    elif mode == Mode.HORIZONTAL_SHRINK:
        carved_img, carved_mask, seam_mask = remove_seam(img, mask, energy_map, axis=0)
    else:
        raise ValueError
    return carved_img, carved_mask, seam_mask


def img_print(img: np.array) -> None:
    colors = ['red', 'green', 'blue']
    for i in range(3):
        print(colors[i])
        pprint(img[:, :, i])


if __name__ == '__main__':
    r = np.array([[1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1],
                  [1, 3, 3, 3, 3, 3, 1],
                  [1, 3, 3, 3, 3, 3, 1],
                  [1, 3, 3, 3, 3, 3, 1]])
    g = np.array([[1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1],
                  [1, 3, 3, 3, 3, 3, 1],
                  [1, 3, 3, 3, 3, 3, 1],
                  [1, 3, 3, 3, 3, 3, 1]])
    b = np.array([[1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1],
                  [1, 3, 3, 3, 3, 3, 1],
                  [1, 3, 3, 3, 3, 3, 1],
                  [1, 3, 3, 3, 3, 3, 1]])

    colored_img = compose(r, g, b)
    img_print(colored_img)
    energy = compute_energy(colored_img)

    removed_seam1, mask1, _ = remove_seam(colored_img, energy, axis=0)
    img_print(removed_seam1)

    removed_seam2, mask2, _ = remove_seam(colored_img, energy, axis=1)
    img_print(removed_seam2)

    expanded_seam1, exp_mask1, _ = expand_seam(colored_img, energy, axis=0)
    img_print(expanded_seam1)

    expanded_seam2, exp_mask2, _ = expand_seam(colored_img, energy, axis=1)
    img_print(expanded_seam2)

    print('Done!')