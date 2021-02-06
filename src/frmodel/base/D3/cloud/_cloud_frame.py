from scipy.interpolate import griddata

from frmodel.base import CONSTS
from frmodel.base.D2 import Frame2D
from abc import ABC, abstractmethod
import numpy as np


class _Cloud3DFrame(ABC):
    
    @abstractmethod
    def data(self, sample_size=None, transformed=True) -> np.ndarray:
        ...

    def to_frame(self,
                 sample_size=None, transformed=True,
                 width=None, height=None,
                 method: CONSTS.INTERP3D = CONSTS.INTERP3D.NEAREST) -> Frame2D:
        """ Converts the Cloud3D into a Frame2D

        :param sample_size: The number of random points to use for interpolation
        :param transformed: Whether to shift axis based on header information
        :param width: Width of resulting image
        :param height: Height of resulting image
        :param method: Method of interpolation, use CONSTS.INTERP3D for various methods
        :return: A Frame2D with Z, R, G, B columns
        """
        # Grab array data and scale it down to desired height and width
        ar = self.data(sample_size, transformed)
        height_range  = np.max(ar[..., 0]) - np.min(ar[..., 0])
        width_range = np.max(ar[..., 1]) - np.min(ar[..., 1])

        if height and not width:
            width = int(width_range / height_range * height)
        elif width and not height:
            height = int(height_range / width_range * width)
        else:
            raise Exception("Frame Height or Width must be specified")

        ar[..., 0] = (ar[..., 0] - np.min(ar[..., 0])) / height_range * height
        ar[..., 1] = (ar[..., 1] - np.min(ar[..., 1])) / width_range * width

        # Create grid to estimate
        grid_x, grid_y = np.mgrid[0:height, 0:width]
        grid = grid_x, grid_y
        method: str
        ar_intp = np.zeros(shape=(height, width, 4), dtype=np.float)

        ar_intp[..., 0] = griddata(ar[..., 0:2], ar[..., 2], grid, method)

        ar_intp[..., 1] = griddata(ar[..., 0:2], ar[..., 3], grid, method)
        ar_intp[..., 2] = griddata(ar[..., 0:2], ar[..., 4], grid, method)
        ar_intp[..., 3] = griddata(ar[..., 0:2], ar[..., 5], grid, method)

        return Frame2D(ar_intp.swapaxes(0, 1), labels=(CONSTS.CHN.Z, *CONSTS.CHN.RGB))

    @staticmethod
    def interp_sig_clamp(x: np.ndarray, alpha: float = 255, beta: float = 50):
        """ Used to clamp the RGB values based on the following formula

                 a
        ------------------
                   x-a/2
        1 + exp( - ----- )
                     b

        :param x: Input
        :param alpha: Amplitude, makes clamping from [0, a]
        :param beta: Bend factor.
        """

        return alpha / (1 + np.exp( - (x - alpha / 2) / beta))


