from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

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
        return self.create(self[top:-bottom or None, left:-right or None, ...],
                           self.labels)

    def crop_glcm(self: 'Frame2D', glcm_radius: int):
        """ Crops the frame to match GLCM cropping. """
        return self.crop(glcm_radius + 1, glcm_radius + 1, glcm_radius, glcm_radius)

    def save(self: 'Frame2D', file_path: str, **kwargs) -> None:
        """ Saves the current Frame file

        :param file_path: Path to save to
        :param kwargs: These kwargs are passed into Image.save(\*\*kwargs)
        """
        Image.fromarray(self.data_rgb().astype(np.uint8)).save(file_path, **kwargs)

    def convolute(self: 'Frame2D', radius: int) -> Frame2D:
        """ Convolutes the Frame, average per window

        :param radius: The radius of the convolution.
        """

        kernel_diam = radius * 2 + 1
        data = np.average(np.average(self.view_windows(kernel_diam, kernel_diam, 1, 1), axis=2), axis=2)
        return self.create(data, self.labels)

    def nanmask(self: 'Frame2D'):
        return np.isnan(self).any(axis=-1)