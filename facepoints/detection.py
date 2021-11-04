import logging
import random
from os import listdir
from os.path import join, isfile
from typing import Tuple, Dict, List, Union, Type, Iterator

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from skimage import io
from skimage.color import gray2rgb
from skimage.transform import rescale
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

Coord = Tuple[int, int]
AnyLayer = Union[nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Linear, nn.BatchNorm2d, nn.Flatten]

logger = logging.Logger(__name__)


def pad_to_square(matrix: NDArray) -> NDArray:
    h, w = matrix.shape
    if h > w:
        padding = ((0, 0), (0, h - w))
    else:
        padding = ((0, w - h), (0, 0))
    return np.pad(matrix, padding)


class Detector(nn.Module):

    _type2class_layer: Dict[str, Type[AnyLayer]] = {
        'conv': nn.Conv2d,
        'relu': nn.ReLU,
        'maxpooling': nn.MaxPool2d,
        'linear': nn.Linear,
        'batchnorm': nn.BatchNorm2d,
        'flatten': nn.Flatten
    }

    def __init__(self, architecture: List[Tuple[str, dict]]):
        super().__init__()
        self._layers: list = [None] * len(architecture)
        for idx, (layer_type, params) in enumerate(architecture):
            layer = self._type2class_layer[layer_type](**params)
            self.__setattr__(self._get_layer_name(layer_type, idx), layer)
            self._layers[idx] = layer

    def forward(self, imgs: Tensor) -> Tensor:
        res = imgs
        for layer in self._layers:
            res: Tensor = layer(res)
        return res

    @staticmethod
    def _get_layer_name(layer_type: str, layer_position: int):
        return 'l' + str(layer_position) + '_' + layer_type


class ImageDirDataset(Dataset[Tuple[Tensor, NDArray, str]]):

    def __init__(self, dirname: str, coords: Dict[str, NDArray] = None, image_size: int = 64, num_coords: int = 28, shuffle: bool = False):
        self._dirname = dirname
        self._image_size = image_size
        self._filenames = [f for f in listdir(dirname) if isfile(join(dirname, f))]
        self._coords = coords.copy() if coords is not None else {
            filename: np.zeros(num_coords) for filename in self._filenames
        }
        self._transformed_coords: Dict[str, NDArray] = {}

        if shuffle:
            random.shuffle(self._filenames)

    def __len__(self) -> int:
        return len(self._filenames)

    def __iter__(self) -> Iterator[Tuple[Tensor, NDArray, str]]:
        yield from zip(map(self._image_reader, self._filenames), map(self._coords_getter, self._filenames), self._filenames)

    def __getitem__(self, index) -> Tuple[Tensor, NDArray, str]:
        img_filename = self._filenames[index]
        return self._image_reader(img_filename), self._coords_getter(img_filename), img_filename

    def _image_reader(self, img_filename: str) -> Tensor:
        filename = join(self._dirname, img_filename)
        img = io.imread(filename)
        if len(img.shape) != 3:
            img = gray2rgb(img)

        red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        padded_red = pad_to_square(red)
        padded_green = pad_to_square(green)
        padded_blue = pad_to_square(blue)

        scale = self._image_size / padded_red.shape[1]
        rescaled_red = rescale(padded_red, scale=scale)
        rescaled_green = rescale(padded_green, scale=scale)
        rescaled_blue = rescale(padded_blue, scale=scale)

        self._transformed_coords[img_filename] = self._coords[img_filename] * scale

        stacked = np.stack([rescaled_red, rescaled_green, rescaled_blue])
        return torch.tensor(stacked, dtype=torch.float, requires_grad=False)

    def _coords_getter(self, img_filename: str) -> NDArray:
        if img_filename in self._transformed_coords:
            return self._transformed_coords[img_filename]
        else:
            logger.warning(f'Reading untransformed coords for image {img_filename}!')
            return self._coords[img_filename]

    @staticmethod
    def collate_fn(items: List[Tuple[Tensor, NDArray, str]]):
        features = []
        coords = []
        filenames = []

        for feature, coord, filename in items:
            features.append(feature)
            coords.append(torch.tensor(coord, dtype=torch.float, requires_grad=False))
            filenames.append(filename)

        features = torch.stack(features)
        coords = torch.stack(coords)

        return features, coords, filenames


def train_detector(true_coords: Dict[str, NDArray], train_data_dir: str, fast_train: bool = False) -> Detector:
    if fast_train:
        device = 'cpu'
        epochs = 1
    else:
        epochs = 10
        device = 'gpu' if torch.cuda.is_available() else 'cpu'

    logger.info(f'Training on {device}')

    batch_size = 256
    img_size = 64
    optim_params = {
        'lr': 0.001,
        'betas': (0.9, 0.999),
        'weight_decay': 0
    }

    train_dataset = ImageDirDataset(train_data_dir, true_coords, image_size=64, shuffle=True)
    train_dataloader = DataLoader(train_dataset, collate_fn=ImageDirDataset.collate_fn, batch_size=batch_size, shuffle=True, num_workers=4)

    model = Detector([
        ('batchnorm', {
            'num_features': 3
        }),
        ('conv', {
            'in_channels': 3,
            'out_channels': 64,
            'kernel_size': (3, 3),
            'padding': 1
        }),
        ('relu', {}),
        ('maxpooling', {
            'kernel_size': (2, 2)
        }),

        ('batchnorm', {
            'num_features': 64
        }),
        ('conv', {
            'in_channels': 64,
            'out_channels': 128,
            'kernel_size': (3, 3),
            'padding': 1
        }),
        ('relu', {}),
        ('maxpooling', {
            'kernel_size': (2, 2)
        }),

        ('batchnorm', {
            'num_features': 128
        }),
        ('conv', {
            'in_channels': 128,
            'out_channels': 256,
            'kernel_size': (3, 3),
            'padding': 1
        }),
        ('relu', {}),
        ('maxpooling', {
            'kernel_size': (2, 2)
        }),

        ('batchnorm', {
            'num_features': 256
        }),
        ('flatten', {}),
        ('linear', {
            'in_features': 256 * ((img_size // 8) ** 2),
            'out_features': 64
        }),
        ('relu', {}),
        ('linear', {
            'in_features': 64,
            'out_features': 28
        })
    ])

    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), **optim_params)

    model.train()
    for epoch in range(epochs):
        losses = []
        for features, coords, filenames in tqdm(train_dataloader, desc=f'Training epoch {epoch}', leave=False):
            optimizer.zero_grad()
            pred_coords = model(features)
            loss: Tensor = loss_func(pred_coords, coords)
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu())

        logger.info(f'{epoch}: [loss]{np.mean(losses)}')

    return model


def detect(model_filename: str, data_dir: str) -> Dict[str, NDArray]:
    pass


def read_coords_from_csv(csv_filename: str) -> Dict[str, NDArray]:
    coord_df: DataFrame = pd.read_csv(csv_filename)
    res = {}
    for idx, row in coord_df.iterrows():
        filename = row['filename']
        row = row.drop('filename')
        res[filename] = row.to_numpy(dtype=int)

    return res


if __name__ == '__main__':
    gold_coords = read_coords_from_csv('public_tests/00_test_img_input/train/gt.csv')
    data_directory = 'public_tests/00_test_img_input/train/images'
    model = train_detector(gold_coords, data_directory)
    torch.save(model.state_dict(), 'facepoints_model.ckpt')