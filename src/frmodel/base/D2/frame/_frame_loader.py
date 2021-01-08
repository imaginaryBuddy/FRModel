from __future__ import annotations

from abc import ABC
from math import ceil
from typing import TYPE_CHECKING

from osgeo import gdal

import numpy as np
from PIL import Image
from skimage.transform import resize

from frmodel.base import CONSTS

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D


class _Frame2DLoader(ABC):

    @classmethod
    def from_image(cls: 'Frame2D', file_path: str, scale:float = 1.0, scale_method=Image.NEAREST) -> _Frame2DLoader:
        """ Creates an instance using the file path.

        :param file_path: Path to image
        :param scale: The scaling to use
        :param scale_method: The method of scaling. See Image.resize

        :returns: Frame2D"""
        img: Image.Image = Image.open(file_path)
        if scale != 1.0:
            img = img.resize([int(scale * s) for s in img.size], resample=scale_method)
        # noinspection PyTypeChecker
        ar = np.asarray(img)[..., :3]

        return cls.create(data=ar, labels=CONSTS.CHN.RGB)

    @classmethod
    def from_nxy_(cls: 'Frame2D', ar: np.ndarray, labels, xy_pos=(3, 4),  width=None, height=None) -> _Frame2DLoader:
        """ Rebuilds the frame with XY values. XY should be of integer values, otherwise, will be casted.

        Note that RGB channels SHOULD be on index 0, 1, 2 else some functions may break. However, can be ignored.

        The frame will be rebuild and all data will be retained, including XY.

        :param ar: The array to rebuild
        :param xy_pos: The positions of X and Y.
        :param labels: The labels of the new Frame2D, excluding XY
        :param height: Height of expected image, if None, Max will be used
        :param width: Width of expected image, if None, Max will be used

        :returns: Frame2D
        """
        max_y = height if height else np.max(ar[:,xy_pos[1]]) + 1
        max_x = width if width else np.max(ar[:,xy_pos[0]]) + 1

        fill = np.zeros(( ceil(max_y), ceil(max_x), ar.shape[-1]), dtype=ar.dtype)

        # Vectorized X, Y <- RGBXY... Assignment
        fill[ar[:, xy_pos[1]].astype(int),
             ar[:, xy_pos[0]].astype(int)] = ar[:]

        return cls.create(data=fill, labels=labels)

    @classmethod
    def from_image_spec(cls: 'Frame2D',
                        file_path_red: str,
                        file_path_green: str,
                        file_path_blue: str,
                        file_path_red_edge: str,
                        file_path_nir: str,
                        scale: float = 1.0) -> _Frame2DLoader:

        labels = [*cls.CHN.RGB, cls.CHN.RED_EDGE, cls.CHN.NIR]

        band_ds: gdal.Dataset = gdal.Open(file_path_red)
        data = np.zeros(shape=(5, band_ds.RasterYSize, band_ds.RasterXSize), dtype=np.float)
        data[0, ...] = band_ds.GetRasterBand(1).ReadAsArray()
        del band_ds

        for e, fp in enumerate((file_path_green, file_path_blue, file_path_red_edge, file_path_nir)):
            band_ds: gdal.Dataset = gdal.Open(fp)
            data[e + 1, ...] = band_ds.GetRasterBand(1).ReadAsArray()
            del band_ds

        data = np.moveaxis(data, 0, -1)
        data = resize(data, output_shape=[int(scale * data.shape[0]),
                                          int(scale * data.shape[1])],
                      order=0)

        return cls.create(data=np.ma.masked_invalid(data, copy=False), labels=labels)
