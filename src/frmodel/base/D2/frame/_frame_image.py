from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from scipy.signal.windows import gaussian
from skimage.transform import rescale

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D


class _Frame2DImage(ABC):
    """ This class handles the transformations like an image editor would have. """

    data: np.ndarray

    def crop(self,
             top:int = 0,
             right:int = 0,
             bottom:int = 0,
             left:int = 0) -> _Frame2DImage:
        """ Crops the frame by specifying how many rows/columns to remove from each side."""
        self: 'Frame2D'
        return self.create(self.data[top:-bottom or None, left:-right or None, ...],
                           self.labels)

    def crop_glcm(self, glcm_radius):
        """ Crops the frame to match GLCM cropping. """
        return self.crop(glcm_radius + 1, glcm_radius + 1, glcm_radius, glcm_radius)

    def save(self, file_path: str, **kwargs) -> None:
        """ Saves the current Frame file """
        self: 'Frame2D'
        Image.fromarray(self.data_rgb().data.astype(np.uint8)).save(file_path, **kwargs)

    def convolute(self, radius: int, method: str = 'nearest') -> Frame2D:
        """ Convolutes the Frame. """

        # We trim the frame so that the new glcm can fit
        # We also average the shifted frame with the current
        kernel_diam = radius * 2 + 1
        if method == 'nearest':
            kernel = np.zeros([kernel_diam + 1, kernel_diam + 1, 1])
            kernel[kernel.shape[0] // 2,
                   kernel.shape[1] // 2] = 1
        else:  # 'average'
            kernel = np.outer(gaussian(kernel_diam + 1, radius),
                              gaussian(kernel_diam + 1, radius))
            kernel = np.expand_dims(kernel, axis=-1)

        self: 'Frame2D'
        return self.create(fftconvolve(self.data, kernel, mode='valid', axes=[0, 1]), self.labels)

    def _rescale(self,
                scale: float,
                dtype=np.uint8,
                anti_aliasing=False) -> _Frame2DImage:
        """ Rescales the image. NOTE THAT THIS WILL RETURN A RGB FRAME ONLY

        Private because it causes weird behavior

        :param scale: The scaling factor. 0.5 for zoom 2x, 2 for 0.5x
        :param dtype: The resulting dtype of the Frame2D
        :param anti_aliasing: Whether to have anti-aliasing or not
        :return: RGB Frame2D
        """

        self: 'Frame2D'
        return self.create(
            data=(np.round
                (rescale
                     (self.data_rgb().data.astype(dtype),
                      scale=scale,
                      anti_aliasing=anti_aliasing,
                      multichannel=True
                      ) * 256)).astype(dtype),
            labels=self.labels)
