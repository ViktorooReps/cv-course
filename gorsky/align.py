from typing import Tuple, Optional, Iterable, List, Callable
import numpy as np

from numpy.fft import irfft2, rfft2
from numpy import conjugate

from enum import IntEnum

from itertools import product


Offset = Tuple[int, int]


class Color(IntEnum):
    RED = 2
    GREEN = 1
    BLUE = 0


class Image(object):
    __slots__ = (
        '_value', '_color', '_offset'
    )

    def __init__(self, value: np.ndarray, color: Color, offset: Offset = (0, 0)):
        self._value = value
        self._color = color
        self._offset = offset

    @property
    def value(self) -> np.ndarray:
        return self._value

    @property
    def color(self) -> Color:
        return self._color

    @property
    def offset(self) -> Offset:
        return self._offset

    def __repr__(self) -> str:
        return str((self._value, self._color, self._offset))

    def realign(self, offset: Offset) -> None:
        self._offset = (offset[0] + self._offset[0], offset[1] + self.offset[1])

    def crop(self, start_coord: Offset, end_coord: Offset) -> None:
        off_st_coord = (start_coord[0] - self._offset[0], start_coord[1] - self._offset[1])
        off_end_coord = (end_coord[0] - self._offset[0], end_coord[1] - self._offset[1])

        self._offset = start_coord

        print(f'Cropping {self._color.name} to {off_st_coord}, {off_end_coord}')

        self._value = self._value[off_st_coord[0]:off_end_coord[0], off_st_coord[1]:off_end_coord[1]]


def combine_images(images: Iterable[Image]) -> np.ndarray:
    images = tuple(images)

    if len(images) != 3:
        raise ValueError
    
    if any((img1.offset != img2.offset) or (img1.value.shape != img2.value.shape) for img1, img2 in product(images, images)):
        raise ValueError

    stack: List[Optional[np.ndarray]] = [None] * 3
    for image in images:
        stack[image.color.value] = image.value

    return np.dstack(stack)


def preprocess(img: np.ndarray) -> Tuple[Image, Image, Image]:   
    blue_crop = img.shape[0] % 3

    part = img.shape[0] // 3
    red = img[:part, :]
    green = img[part:2 * part, :]
    if not blue_crop:
        blue = img[2 * part:, :]
    else:
        blue = img[2 * part:-blue_crop, :]

    border = img.shape[1] // 20
    if border:
        red = red[border:-border, border:-border]
        green = green[border:-border, border:-border]
        blue = blue[border:-border, border:-border]

    return Image(red, Color.RED), Image(green, Color.GREEN), Image(blue, Color.BLUE)


def fourier_offset(img1: np.ndarray, img2: np.ndarray) -> Offset:
    row_offset, col_offset = img1.shape[0] // 20, img1.shape[1] // 20
    C = irfft2(np.multiply(rfft2(img1), np.conjugate(rfft2(img2))))
    preds = np.unravel_index(np.argmax(C, axis=None), C.shape)
    
    alt_preds: Offset = preds[0] - img1.shape[0], preds[1] - img1.shape[1] 
    true_pred_0 = alt_preds[0] if abs(alt_preds[0]) < preds[0] else preds[0]
    true_pred_1 = alt_preds[1] if abs(alt_preds[1]) < preds[1] else preds[1]

    return true_pred_0, true_pred_1


def choose_offset(images: Iterable[Image], key: Callable = max) -> Offset:
    best_offset: Optional[Offset] = None
    for image in images:
        if best_offset is None:
            best_offset = image.offset
        else:
            best_offset = (key(best_offset[0], image.offset[0]), key(best_offset[1], image.offset[1]))
    
    if best_offset is None:
        raise ValueError
    else:
        return best_offset


def align(img: np.ndarray, coords: Offset) -> Tuple[np.ndarray, Offset, Offset]:
    height = img.shape[0]
    red_x = 0
    green_x = height // 3
    blue_x = (height // 3) * 2

    # offset of images is 5% right now
    imgr, imgg, imgb = preprocess(img)

    # images offset from green image
    off_gr = fourier_offset(imgg.value, imgr.value)
    off_gb = fourier_offset(imgg.value, imgb.value)

    print(f'Calculated red offset: {off_gr}')
    print(f'Calculated blue offset: {off_gb}')

    r_total_off = (red_x - off_gr[0], -off_gr[1])
    b_total_off = (blue_x - off_gb[0], -off_gb[1])

    r_coord = (coords[0] - green_x + r_total_off[0], coords[1] + r_total_off[1])
    b_coord = (coords[0] - green_x + b_total_off[0], coords[1] + b_total_off[1])

    imgr.realign(off_gr)
    imgb.realign(off_gb)

    images = [imgr, imgg, imgb]

    max_off = choose_offset(images, key=max)
    min_off = choose_offset(images, key=min)

    min_off = (min_off[0] + imgg.value.shape[0], min_off[1] + imgg.value.shape[1])

    for image in images:
        image.crop(max_off, min_off)

    combined = combine_images(images)
    return combined, r_coord, b_coord


if __name__ == '__main__':
    red = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    green = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    blue = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]

    img = np.concatenate([np.array(red), np.array(green), np.array(blue)], axis=0)
    align(img, (20, 5))