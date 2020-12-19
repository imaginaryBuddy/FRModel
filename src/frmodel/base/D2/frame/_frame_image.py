from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from scipy.signal.windows import gaussian

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D


class _Frame2DImage(ABC):
    """ This class handles the transformations like an image editor would have. """

    def crop(self: 'Frame2D',
             top:int = 0,
             right:int = 0,
             bottom:int = 0,
             left:int = 0) -> _Frame2DImage:
        """ Crops the frame by specifying how many rows/columns to remove from each side."""
        return self.create(self.data[top:-bottom or None, left:-right or None, ...],
                           self.labels)

    def crop_glcm(self: 'Frame2D', glcm_radius: int):
        """ Crops the frame to match GLCM cropping. """
        return self.crop(glcm_radius + 1, glcm_radius + 1, glcm_radius, glcm_radius)

    def save(self: 'Frame2D', file_path: str, **kwargs) -> None:
        """ Saves the current Frame file

        :param file_path: Path to save to
        :param kwargs: These kwargs are passed into Image.save(\*\*kwargs)
        """
        Image.fromarray(self.data_rgb().data.astype(np.uint8)).save(file_path, **kwargs)

    def convolute(self: 'Frame2D', radius: int, method: str = 'nearest') -> Frame2D:
        """ Convolutes the Frame.

        :param radius: The radius of the convolution.
        :param method: "nearest" or "average". If argument is not 'nearest', it'll use average by default.
        """

        kernel_diam = radius * 2 + 1

        if method == 'nearest':
            kernel = np.zeros([kernel_diam + 1, kernel_diam + 1, 1])
            kernel[kernel.shape[0] // 2,
                   kernel.shape[1] // 2] = 1

        else:  # 'average'
            kernel = np.outer(gaussian(kernel_diam + 1, radius),
                              gaussian(kernel_diam + 1, radius))
            kernel = np.expand_dims(kernel, axis=-1)

        return self.create(fftconvolve(self.data, kernel, mode='valid', axes=[0, 1]), self.labels)
