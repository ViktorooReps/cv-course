from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray
from numpy.fft import fft, ifft, rfft, rfft2, fft2, ifft2


def gaussian_kernel(size: int, sigma: float) -> NDArray:
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра (нечетный)
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    center = size // 2
    coef = 2 * sigma * sigma

    def gaussian_filter(x: int, y: int) -> float:
        r = (x - center) ** 2 + (y - center) ** 2
        return np.exp(- r / coef) / coef / np.pi

    kernel: NDArray = np.fromfunction(gaussian_filter, (size, size), dtype=float)
    return kernel / np.sum(kernel)


def _calculate_pads(shape: Tuple[int, int], target: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    x_pad = target[0] - shape[0]
    y_pad = target[1] - shape[1]

    right_pad_x = x_pad
    right_pad_y = y_pad

    return (0, right_pad_x), (0, right_pad_y)


def fourier_transform(h: NDArray, shape: List[int]) -> NDArray:
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    pads = _calculate_pads(h.shape, shape)
    padded_h = np.pad(h, pads).astype(float)
    return fft2(padded_h)


def inverse_kernel(H: NDArray, threshold: float = 1e-10) -> NDArray:
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    mask = np.abs(H) <= threshold
    return np.divide(1, H, out=np.zeros_like(H), where=~mask)


def inverse_filtering(blurred_img: NDArray, h: NDArray, threshold: float = 1e-10) -> NDArray:
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    G = fft2(blurred_img)
    H = fourier_transform(h, G.shape)
    F_est = G * inverse_kernel(H, threshold=threshold)

    f_est = ifft2(F_est)
    return np.abs(f_est)


def wiener_filtering(blurred_img: NDArray, h: NDArray, K: float = 0.00006) -> NDArray:
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    H = fourier_transform(h, blurred_img.shape)
    H_conj = np.conjugate(H)
    H_sq = H_conj * H

    F_est = (H_conj / (H_sq + K)) * fft2(blurred_img)
    f_est = ifft2(F_est)

    return np.abs(f_est)


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    mse = np.sum((img1 - img2) ** 2) / (img1.shape[0] * img1.shape[1])
    return 20 * np.log10(255 / np.sqrt(mse))
