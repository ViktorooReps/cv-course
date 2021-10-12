import os

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage.filters import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!

# :)
from numpy.typing import NDArray
from typing import Tuple, List, Iterator, Any
from functools import partial  # ну простите не удержался


def pca_compression(matrix: NDArray, p: int) -> Tuple[NDArray, NDArray, NDArray]:
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы и проекция матрицы на новое пр-во
    """
    
    # Your code here
    
    # Отцентруем каждую строчку матрицы
    row_means = np.mean(matrix, axis=1)
    row_centered_matrix = matrix - row_means[:, None]

    # Найдем матрицу ковариации
    cov_matrix = np.cov(matrix)

    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eig_values, eig_vectors = np.linalg.eigh(cov_matrix)

    # Посчитаем количество найденных собственных векторов
    eig_vectors_num = eig_vectors.shape[1]

    # Сортируем собственные значения в порядке убывания
    sorted_eig_values_idxs = np.flip(np.argsort(eig_values))
    sorted_eig_values = eig_values[sorted_eig_values_idxs]

    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    sorted_eig_vectors = eig_vectors[:, sorted_eig_values_idxs]

    # Оставляем только p собственных векторов
    top_p_eig_vectors = sorted_eig_vectors[:, :p]

    # Проекция данных на новое пространство
    data_projection = np.dot(top_p_eig_vectors.T, row_centered_matrix)
    
    return top_p_eig_vectors, data_projection, row_means


def pca_decompression(compressed: List[Tuple[NDArray, NDArray, NDArray]]) -> NDArray:
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """
    
    result_img = []
    for i, (eig_vectors, data_projection, row_means) in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!

        result_img.append(eig_vectors @ data_projection + row_means[:, None])

    result_img = np.dstack(result_img)
    return byte_conv(result_img)


def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        r, g, b = decompose(img)
        compressed_r = pca_compression(r, p)
        compressed_g = pca_compression(g, p)
        compressed_b = pca_compression(b, p)

        decompressed = pca_decompression([compressed_r, compressed_g, compressed_b])
            
        axes[i // 3, i % 3].imshow(decompressed)
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")


def decompose(img: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    return img[:, :, 0].astype(np.float64), img[:, :, 1].astype(np.float64), img[:, :, 2].astype(np.float64)


def compose(red: NDArray, green: np.array, blue: np.array) -> NDArray:
    return byte_conv(np.dstack([red, green, blue]))


def byte_conv(img: NDArray) -> NDArray:
    min_mask = img > 0
    max_mask = img <= 255
    inv_max_mask = img > 255
    return (img * min_mask * max_mask + inv_max_mask * 255).astype(np.uint8)


def rgb2ycbcr(img: NDArray) -> NDArray:
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """
    bias = np.array([0, 128, 128], dtype=float)
    transform = np.array([[ 0.299,   0.587,   0.114],
                          [-0.1687, -0.3313,  0.5],
                          [ 0.5,    -0.4187, -0.0813]], dtype=float)
    ycbcr = bias + img.astype(float) @ transform.T
    return byte_conv(ycbcr)


def ycbcr2rgb(img):
    """ Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """
    bias = np.array([0, -128, -128], dtype=float)
    transform = np.array([[1,  0,        1.402],
                          [1, -0.34414, -0.71414],
                          [1,  1.77,     0]], dtype=float)
    rgb = (img.astype(float) + bias) @ transform.T
    return byte_conv(rgb)


def get_gauss_1():
    plt.clf()
    rgb_img: NDArray = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ycbcr_img = rgb2ycbcr(rgb_img)
    y, cb, cr = decompose(ycbcr_img)
    blurred_cb = gaussian_filter(cb, sigma=1)
    blurred_cr = gaussian_filter(cr, sigma=1)

    blurred_ycbcr_img = compose(y, blurred_cb, blurred_cr)
    blurred_rgb = ycbcr2rgb(blurred_ycbcr_img)

    plt.imshow(blurred_rgb)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    ycbcr_img = rgb2ycbcr(rgb_img)
    y, cb, cr = decompose(ycbcr_img)
    blurred_y = gaussian_filter(y, sigma=1)

    blurred_ycbcr_img = compose(blurred_y, cb, cr)
    blurred_rgb = ycbcr2rgb(blurred_ycbcr_img)

    plt.imshow(blurred_rgb)
    plt.savefig("gauss_2.png")


def downsampling(component: NDArray) -> NDArray:
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """
    
    return gaussian_filter(component, sigma=10)[::2, ::2]


def dct(block: NDArray) -> NDArray:
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

    def alpha(idx: int) -> float:
        return 1 / np.sqrt(2) if idx == 0 else 1

    res = np.zeros(block.shape, dtype=float)

    # god forgive me for I have committed a sin of cycle
    for u in range(block.shape[0]):
        for v in range(block.shape[1]):
            def cos_coeff(xy: NDArray, uv: float) -> NDArray:
                return np.cos((2 * xy + 1) * uv * np.pi / 16)

            def cos_matr(x: NDArray, y: NDArray) -> NDArray:
                return cos_coeff(x, u) * cos_coeff(y, v)

            res[u, v] = (1 / 4) * alpha(u) * alpha(v) * np.sum(block * np.fromfunction(cos_matr, block.shape))

    return res


# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block: NDArray, quantization_matrix: NDArray) -> NDArray:
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """
    
    return np.round(block.astype(float) / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix: NDArray, q: float):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    if 1 <= q < 50:
        scale = 5000 / q
    elif 50 <= q <= 99:
        scale = 200 - 2 * q
    else:
        scale = 1

    quantization_matrix = np.floor((50 + scale * default_quantization_matrix.astype(float)) / 100)

    return quantization_matrix + (quantization_matrix == 0)


def zigzag_gen(max_idx: Tuple[int, int]) -> Iterator[Tuple[int, int]]:
    coord = 0, 0
    yield coord

    def move(crd: Tuple[int, int]) -> Tuple[int, int]:
        if crd[0] == 0 or crd[0] == max_idx[0]:
            return crd[0], crd[1] + 1
        if crd[1] == 0 or crd[1] == max_idx[1]:
            return crd[0] + 1, crd[1]
        raise ValueError

    def ascend(crd: Tuple[int, int]) -> Iterator[Tuple[int, int]]:
        while crd[0] != 0 and crd[1] != max_idx[1]:
            crd = crd[0] - 1, crd[1] + 1
            yield crd

    def descend(crd: Tuple[int, int]) -> Iterator[Tuple[int, int]]:
        while crd[1] != 0 and crd[0] != max_idx[0]:
            crd = crd[0] + 1, crd[1] - 1
            yield crd

    while coord != max_idx:
        coord = move(coord)
        yield coord
        if coord == max_idx:
            break

        for c in descend(coord):
            coord = c
            yield coord

        coord = move(coord)
        yield coord
        if coord == max_idx:
            break

        for c in ascend(coord):
            coord = c
            yield coord


def zigzag(block: NDArray) -> List:
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """

    max_idx = block.shape[0] - 1, block.shape[1] - 1

    return [block[coord] for coord in zigzag_gen(max_idx)]


def compression(zigzag_list: List) -> NDArray:
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """

    def zero_compressor(lst: List) -> Iterator:
        cnt = 0
        for item in lst:
            if item == 0:
                cnt += 1
                continue

            if cnt != 0:
                yield 0
                yield cnt
                cnt = 0
            yield item

        if cnt != 0:
            yield 0
            yield cnt

    return np.array(list(zero_compressor(zigzag_list)))


def jpeg_compression(img: NDArray, quantization_matrixes: Tuple[NDArray, NDArray]) -> List:
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Your code here
    
    # Переходим из RGB в YCbCr

    ycbcr_img = rgb2ycbcr(img)
    y, cb, cr = decompose(ycbcr_img)

    # Уменьшаем цветовые компоненты

    cb_downsampled = downsampling(cb)
    cr_downsampled = downsampling(cr)

    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]

    def block_generator(matr: NDArray) -> Iterator[NDArray]:
        block_size = 8
        for i in range(matr.shape[0] // block_size):
            for j in range(matr.shape[1] // block_size):
                yield matr[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size].astype(float) - 128

    y_blocks = block_generator(y)
    cb_blocks = block_generator(cb_downsampled)
    cr_blocks = block_generator(cr_downsampled)

    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие

    transformed_y_blocks = map(dct, y_blocks)
    transformed_cb_blocks = map(dct, cb_blocks)
    transformed_cr_blocks = map(dct, cr_blocks)

    quant_y_blocks = map(partial(quantization, quantization_matrix=quantization_matrixes[0]), transformed_y_blocks)
    quant_cb_blocks = map(partial(quantization, quantization_matrix=quantization_matrixes[1]), transformed_cb_blocks)
    quant_cr_blocks = map(partial(quantization, quantization_matrix=quantization_matrixes[1]), transformed_cr_blocks)

    zigzagged_y_blocks = map(zigzag, quant_y_blocks)
    zigzagged_cb_blocks = map(zigzag, quant_cb_blocks)
    zigzagged_cr_blocks = map(zigzag, quant_cr_blocks)

    compressed_y_blocks = map(compression, zigzagged_y_blocks)
    compressed_cb_blocks = map(compression, zigzagged_cb_blocks)
    compressed_cr_blocks = map(compression, zigzagged_cr_blocks)

    return list(map(list, [compressed_y_blocks, compressed_cb_blocks, compressed_cr_blocks]))


def inverse_compression(compressed_list: List) -> List:
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """
    
    def zero_decompressor(lst: List) -> Iterator:
        def repeat_zero(times: Any):
            for _ in range(int(times)):
                yield 0

        prev_zero = False
        for item in lst:
            if item == 0:
                prev_zero = True
                continue

            if prev_zero:
                yield from repeat_zero(item)
            else:
                yield item
            prev_zero = False

    return list(zero_decompressor(compressed_list))


def inverse_zigzag(input: List) -> NDArray:
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    max_idx = (7, 7)
    res = np.zeros(shape=[8, 8], dtype=float)
    for coord, item in zip(zigzag_gen(max_idx), input):
        res[coord] = item
    
    return res


def inverse_quantization(block: NDArray, quantization_matrix: NDArray) -> NDArray:
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """
    
    return block * quantization_matrix


def inverse_dct(block: NDArray) -> NDArray:
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    def alpha(idxes: NDArray) -> NDArray:
        return (1 / np.sqrt(2)) * (idxes == 0) + (idxes != 0)

    res = np.zeros(block.shape, dtype=float)

    # god forgive me for I have committed a sin of cycle
    for x in range(block.shape[0]):
        for y in range(block.shape[1]):
            def cos_coeff(xy: float, uv: NDArray) -> NDArray:
                return np.cos((2 * xy + 1) * uv * np.pi / 16)

            def cos_matr(u: NDArray, v: NDArray) -> NDArray:
                return alpha(u) * alpha(v) * cos_coeff(x, u) * cos_coeff(y, v)

            res[x, y] = (1 / 4) * np.sum(block * np.fromfunction(cos_matr, block.shape))

    return np.round(res)


def upsampling(component: NDArray) -> NDArray:
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """
    
    new_shape = 2 * component.shape[0], 2 * component.shape[1]

    res = np.zeros(new_shape, dtype=float)
    mask = np.fromfunction(lambda i, j: ((1 - i % 2) + (1 - j % 2)) == 0, new_shape)
    res[mask] = component.ravel()
    res[np.roll(mask, 1, axis=0)] = component.ravel()
    res[np.roll(mask, 1, axis=1)] = component.ravel()
    res[np.roll(mask, (1, 1), axis=(0, 1))] = component.ravel()

    return res


def jpeg_decompression(result: List, result_shape: Tuple, quantization_matrixes: Tuple[NDArray, NDArray]) -> NDArray:
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """
    result_shape = result_shape[0], result_shape[1]
    downsampled_shape = result_shape[0] // 2, result_shape[1] // 2

    compressed_y_blocks = result[0]
    compressed_cb_blocks = result[1]
    compressed_cr_blocks = result[2]

    zigzagged_y_blocks = map(inverse_compression, compressed_y_blocks)
    zigzagged_cb_blocks = map(inverse_compression, compressed_cb_blocks)
    zigzagged_cr_blocks = map(inverse_compression, compressed_cr_blocks)

    quant_y_blocks = map(inverse_zigzag, zigzagged_y_blocks)
    quant_cb_blocks = map(inverse_zigzag, zigzagged_cb_blocks)
    quant_cr_blocks = map(inverse_zigzag, zigzagged_cr_blocks)

    transformed_y_blocks = map(partial(inverse_quantization, quantization_matrix=quantization_matrixes[0]), quant_y_blocks)
    transformed_cb_blocks = map(partial(inverse_quantization, quantization_matrix=quantization_matrixes[1]), quant_cb_blocks)
    transformed_cr_blocks = map(partial(inverse_quantization, quantization_matrix=quantization_matrixes[1]), quant_cr_blocks)

    y_blocks = map(inverse_dct, transformed_y_blocks)
    cb_blocks = map(inverse_dct, transformed_cb_blocks)
    cr_blocks = map(inverse_dct, transformed_cr_blocks)

    def reshape_blocks(blocks: List, target_shape: Tuple[int, int]) -> List[List]:
        block_size = 8
        row_blocks = target_shape[1] // block_size
        return [blocks[i * row_blocks:(i + 1) * row_blocks] for i in range(len(blocks) // row_blocks)]

    y = np.block(reshape_blocks(list(y_blocks), result_shape)) + 128
    cb_downsampled = np.block(reshape_blocks(list(cb_blocks), downsampled_shape)) + 128
    cr_downsampled = np.block(reshape_blocks(list(cr_blocks), downsampled_shape)) + 128

    cb = upsampling(cb_downsampled)
    cr = upsampling(cr_downsampled)

    ycbcr_img = compose(y, cb, cr)
    rgb_img = ycbcr2rgb(ycbcr_img)

    return rgb_img


def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        y_quant_matrix = own_quantization_matrix(y_quantization_matrix, p)
        color_quant_matrix = own_quantization_matrix(color_quantization_matrix, p)

        compressed_img = jpeg_compression(img, (y_quant_matrix, color_quant_matrix))
        decompressed_img = jpeg_decompression(compressed_img, img.shape, (y_quant_matrix, color_quant_matrix))

        axes[i // 3, i % 3].imshow(decompressed_img)
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg'; 
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]
        
        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))
            
        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])
        
    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')
        
    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), compressed)
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))
        
    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]
    
    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))
     
    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    
    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')
    
    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == '__main__':
    # get_gauss_1()
    # get_gauss_2()
    get_jpeg_metrics_graph()
    # get_pca_metrics_graph()
    jpeg_visualize()
    # pca_visualize()
