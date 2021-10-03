import numpy as np
import numpy.ma as ma

from pprint import pprint
from typing import List, Iterable, Tuple

from itertools import product


def _make_pattern(pattern: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    row_reps = n_rows // pattern.shape[0]
    col_reps = n_cols // pattern.shape[1]

    row_subpattern = n_rows % pattern.shape[0]
    col_subpattern = n_cols % pattern.shape[1]

    pattern_row = np.tile(pattern, (1, col_reps))
    added_pattern = pattern[:, :col_subpattern]
    pattern_row = np.concatenate([pattern_row, added_pattern], axis=1)

    full_pattern = np.tile(pattern_row, (row_reps, 1))
    added_pattern = pattern_row[:row_subpattern, :]
    full_pattern = np.concatenate([full_pattern, added_pattern], axis=0)

    return full_pattern


def get_bayer_masks(n_rows: int, n_cols: int) -> np.ndarray:
    red_pattern = np.array([[0, 1], [0, 0]], dtype=bool)
    green_pattern = np.array([[1, 0], [0, 1]], dtype=bool)
    blue_pattern = np.array([[0, 0], [1, 0]], dtype=bool)

    red = _make_pattern(red_pattern, n_rows, n_cols)
    green = _make_pattern(green_pattern, n_rows, n_cols)
    blue = _make_pattern(blue_pattern, n_rows, n_cols)

    return np.dstack([red, green, blue])


def get_colored_img(img: np.ndarray) -> np.ndarray:
    colored_img = np.dstack([img, img, img])
    masked: ma.masked_array = ma.masked_array(colored_img, mask=~get_bayer_masks(*img.shape), dtype=np.uint8)
    return masked.filled(0)


def _sum_shifted(img: np.ndarray, shifts: Iterable[Tuple[int, int]], coeffs: Iterable[float] = None) -> np.ndarray:
    shifts = tuple(shifts)
    if coeffs is None:
        coeffs = [1] * len(shifts)

    max_shift = max(max(abs(s[0]), abs(s[1])) for s in shifts)
    pad = np.array([[max_shift, max_shift], [max_shift, max_shift]])
    shifted = []
    for (row_shift, col_shift), coeff in zip(shifts, coeffs):
        shift = np.array([[-col_shift, col_shift], [-row_shift, row_shift]])
        curr_pad = pad - shift
        padded = np.pad(img, curr_pad)
        shifted.append(padded.astype(float) * coeff)
    
    summed = np.sum(np.dstack(shifted), axis=2)
    return summed[max_shift:-max_shift, max_shift:-max_shift]


def _sum_neighbors(img: np.ndarray, num: int = 1) -> np.ndarray:
    shifts = list(product(range(-num, num + 1), range(-num, num + 1)))
    return _sum_shifted(img, shifts).astype(int)


def bilinear_interpolation(colored_img: np.ndarray) -> np.ndarray:
    masks = get_bayer_masks(colored_img.shape[0], colored_img.shape[1])
    
    interp_img: List[np.ndarray] = []
    for layer in range(3):
        msk = masks[:, :, layer].astype(np.uint8)
        img = colored_img[:, :, layer].astype(np.uint8)
        interp_img.append((_sum_neighbors(img) // _sum_neighbors(msk)).astype(np.uint8))
    
    inv_mask = (~masks.astype(bool)).astype(np.uint8)
    combined: np.ndarray = np.dstack(interp_img) * inv_mask + colored_img
    return combined


def improved_interpolation(raw_img: np.ndarray) -> np.ndarray:
    masks = get_bayer_masks(raw_img.shape[0], raw_img.shape[1])
    red_mask = masks[:, :, 0]
    green_mask = masks[:, :, 1]
    blue_mask = masks[:, :, 2]

    colored_img = get_colored_img(raw_img)

    red_raw = colored_img[:, :, 0].astype(float)
    green_raw = colored_img[:, :, 1].astype(float)
    blue_raw = colored_img[:, :, 2].astype(float)

    # interpolating green channel

    rnb_raw = red_raw + blue_raw
    green_est = 4 * rnb_raw  # central blue/red cell with 4 coeff

    cross_shifts = [(0, -2), (0, 2), (-2, 0), (2, 0)] 
    green_est += -1 * _sum_shifted(rnb_raw, cross_shifts)  # add blue/red cells with -1 coeff

    cross_shifts = [(0, -1), (0, 1), (-1, 0), (1, 0)] 
    green_est += 2 * _sum_shifted(green_raw, cross_shifts)  # add green cells with 2 coeff

    green_norm = (red_mask + blue_mask) * green_est / 8  # apply combined mask of red and blue channels
    final_green = np.clip(green_raw + green_norm, a_max=255, a_min=0).astype(np.uint8)  # combine estimations with known values

    # interpolating blue/red at green (common part)

    rnb_est = 5 * green_raw  # central green with 5 coeff

    corners_shifts = [(1, 1), (-1, -1), (-1, 1), (1, -1)]
    rnb_est += -1 * _sum_shifted(green_raw, corners_shifts)  # add corner green cells with -1 coeff

    padded_green_mask = np.pad(green_mask, 1)
    shifted_blue_mask = np.pad(blue_mask, ((1, 1), (2, 0)))
    shifted_red_mask = np.pad(red_mask, ((1, 1), (0, 2)))
    blue_row_red_column_green_mask = (padded_green_mask & shifted_blue_mask)[1:-1, 1:-1]
    red_row_blue_column_green_mask = (padded_green_mask & shifted_red_mask)[1:-1, 1:-1]
    
    # summing up two different situations for each channel on green

    cross_shifts = [(0, -2), (0, 2), (-2, 0), (2, 0)] 
    rrbc_red_coefs = [1/2, 1/2, -1, -1]
    rrbc_blue_coefs = [-1, -1, 1/2, 1/2]
    brrc_red_coefs = [-1, -1, 1/2, 1/2]
    brrc_blue_coefs = [1/2, 1/2, -1, -1]

    # add greens dependant on situation
    red_on_green_est = red_row_blue_column_green_mask * _sum_shifted(green_raw, cross_shifts, rrbc_red_coefs)
    red_on_green_est += blue_row_red_column_green_mask * _sum_shifted(green_raw, cross_shifts, brrc_red_coefs)

    # add neighboring red cells
    red_on_green_est += red_row_blue_column_green_mask * _sum_shifted(red_raw, [(1, 0), (-1, 0)], [4, 4])
    red_on_green_est += blue_row_red_column_green_mask * _sum_shifted(red_raw, [(0, 1), (0, -1)], [4, 4])

    # add greens dependant on situation
    blue_on_green_est = red_row_blue_column_green_mask * _sum_shifted(green_raw, cross_shifts, rrbc_blue_coefs)
    blue_on_green_est += blue_row_red_column_green_mask * _sum_shifted(green_raw, cross_shifts, brrc_blue_coefs)
    
    # add neighboring blue cells
    blue_on_green_est += blue_row_red_column_green_mask * _sum_shifted(blue_raw, [(1, 0), (-1, 0)], [4, 4])
    blue_on_green_est += red_row_blue_column_green_mask * _sum_shifted(blue_raw, [(0, 1), (0, -1)], [4, 4])

    red_on_green_est = (rnb_est + red_on_green_est) * green_mask 
    blue_on_green_est = (rnb_est + blue_on_green_est) * green_mask 

    # interpolating red on blue and blue on red:

    corners_coeffs = [2, 2, 2, 2]
    corners_shifts = [(1, 1), (-1, -1), (-1, 1), (1, -1)]
    cross_coeffs = [-3/2, -3/2, -3/2, -3/2]
    cross_shifts = [(0, -2), (0, 2), (-2, 0), (2, 0)]
    rnb_est = 6 * rnb_raw + _sum_shifted(rnb_raw, [*corners_shifts, *cross_shifts], [*corners_coeffs, *cross_coeffs])
    
    red_norm = (red_on_green_est + rnb_est * blue_mask) / 8 + red_raw
    blue_norm = (blue_on_green_est + rnb_est * red_mask) / 8 + blue_raw

    final_red = (np.clip(red_norm, a_max=255, a_min=0)).astype(np.uint8)
    final_blue = (np.clip(blue_norm, a_max=255, a_min=0)).astype(np.uint8)
    
    return np.dstack([final_red, final_green, final_blue])


def compute_psnr(img_pred: np.ndarray, img_gt: np.ndarray) -> float:
    assert img_pred.shape == img_gt.shape
    shape = img_gt.shape 

    img_pred = img_pred.astype(float)
    img_gt = img_gt.astype(float)

    chw = shape[0] * shape[1] * shape[2]
    mse = (1 / chw) * np.sum((img_pred - img_gt) ** 2)

    if not mse:
        raise ValueError

    sqmax = np.max(img_gt ** 2)
    return 10 * np.log10(sqmax / mse)


def img_print(img):
    colors = ['red', 'green', 'blue']
    for i in range(3):
        print(colors[i])
        pprint(img[:, :, i])


if __name__ == '__main__':
    arr = np.array([
        [1, 1, 2, 1, 1],
        [1, 2, 3, 2, 1], 
        [4, 5, 6, 5, 4], 
        [7, 8, 9, 8, 7]
    ], dtype=np.uint8)
    print('arr', arr.shape)
    pprint(arr)
    print('intrp')
    img_print(improved_interpolation(arr))
