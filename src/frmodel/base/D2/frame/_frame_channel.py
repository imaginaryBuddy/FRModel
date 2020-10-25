from __future__ import annotations

from abc import abstractmethod

import numpy as np
from scipy.signal import fftconvolve
from scipy.signal.windows import gaussian
from skimage.color import rgb2hsv

from frmodel.base.D2.frame._frame_channel_glcm import _Frame2DChannelGLCM
from frmodel.base.consts import CONSTS

CHANNEL = CONSTS.CHANNEL
MAX_RGB = 255


class _Frame2DChannel(_Frame2DChannelGLCM):

    data: np.ndarray
    
    @abstractmethod
    def data_rgb(self): ...
    
    @abstractmethod
    def data_chn(self, *args, **kwargs): ...

    @abstractmethod
    def width(self): ...

    @abstractmethod
    def height(self): ...
    
    @abstractmethod
    def slide_xy(self, *args, **kwargs): ...

    # noinspection PyArgumentList
    @classmethod
    def init(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def get_hsv(self) -> np.ndarray:
        """ Creates a HSV """
        return rgb2hsv(self.data_rgb())

    def get_ex_g(self, modified=False) -> np.ndarray:
        """ Calculates the excessive green index

        Original: 2g - 1r - 1b
        Modified: 1.262g - 0.884r - 0.331b

        :param modified: Whether to use the modified or not. Refer to docstring
        """

        if modified:
            return 1.262 * self.data_chn(CHANNEL.RED) - \
                   0.884 * self.data_chn(CHANNEL.GREEN) - \
                   0.331 * self.data_chn(CHANNEL.BLUE)

        else:
            return 2 * self.data_chn(CHANNEL.RED) - \
                   self.data_chn(CHANNEL.GREEN) - \
                   self.data_chn(CHANNEL.BLUE)

    def get_ex_gr(self) -> np.ndarray:
        """ Calculates the excessive green minus excess red index

        2g - r - b - 1.4r + g = 3g - 2.4r - b
        """

        return 3 * self.data_chn(CHANNEL.RED) - 2.4 * self.data_chn(CHANNEL.GREEN) - self.data_chn(CHANNEL.BLUE)

    def get_ndi(self):
        """ Calculates the Normalized Difference Index

        (g - r) / (g + r)
        """

        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.nan_to_num(
                np.true_divide(self.data_chn(CHANNEL.GREEN).astype(np.int) -
                               self.data_chn(CHANNEL.RED).astype(np.int),
                               self.data_chn(CHANNEL.GREEN).astype(np.int) +
                               self.data_chn(CHANNEL.RED).astype(np.int)),
                copy=False, nan=0, neginf=0, posinf=0)

        return x

    def get_veg(self, const_a: float = 0.667):
        """ Calculates the Vegetative Index

        g / {(r^a) * [b^(1 - a)]}

        :param const_a: Constant A depends on the camera used.
        """

        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.nan_to_num(self.data_chn(CHANNEL.GREEN).astype(np.float) /
                              (np.power(self.data_chn(CHANNEL.RED).astype(np.float), const_a) *
                               np.power(self.data_chn(CHANNEL.BLUE).astype(np.float), 1 - const_a)),
                              copy=False, nan=0, neginf=0, posinf=0)
        return x

    def get_xy(self) -> np.ndarray:
        """ Creates the XY Coord Array """

        # We create a new array to copy self over we expand the last axis by 2
        buffer = np.zeros([*self.data.shape[0:-1], 2])

        # Create X & Y then copy over
        buffer[..., 0] = np.tile(np.arange(0, self.width()), (self.height(), 1))
        buffer[..., 1] = np.tile(np.arange(0, self.height()), (self.width(), 1)).swapaxes(0, 1)

        return buffer

    def get_chns(self, self_=False, xy=False, hsv=False, ex_g=False,
                 mex_g=False, ex_gr=False, ndi=False, veg=False,
                 veg_a=0.667, glcm_con=False, glcm_cor=False, glcm_ent=False,
                 glcm_by_x=1, glcm_by_y=1,
                 glcm_radius=5, glcm_verbose=False,
                 conv_gauss_stdev=4) -> _Frame2DChannel:
        """ Gets selected channels

        Order is given by the argument order.
        R, G, B, X, Y, H, S, V, EX_G, MEX_G, EX_GR, NDI, VEG,
        ConR, ConG, ConB, CorrR, CorrG, CorrB, EntR, EntG, EntB

        :param self_: Include current frame
        :param xy: XY Coordinates
        :param hsv: Hue, Saturation, Value
        :param ex_g: Excess Green
        :param mex_g: Modified Excess Green
        :param ex_gr: Excess Green Minus Red
        :param ndi: Normalized Difference Index
        :param veg: Vegetative Index
        :param veg_a: Vegatative Index Const A
        :param glcm_con: GLCM Contrast
        :param glcm_cor: GLCM Correlation
        :param glcm_ent: GLCM Entropy
        :param glcm_by_x: GLCM By X Parameter
        :param glcm_by_y: GLCM By Y Parameter
        :param glcm_radius: GLCM Radius
        :param glcm_verbose: Whether to have glcm entropy generation give feedback
        :param conv_gauss_stdev: The stdev of the gaussian kernel used to convolute existing channels if GLCM is used
        """
        return self.get_all_chns(self_, xy, hsv, ex_g, mex_g, ex_gr, ndi,
                                 veg, veg_a, glcm_con, glcm_cor, glcm_ent,
                                 glcm_by_x, glcm_by_y, glcm_radius, glcm_verbose,
                                 conv_gauss_stdev)

    def get_all_chns(self, self_=True, xy=True, hsv=True, ex_g=True, mex_g=True,
                     ex_gr=True, ndi=True, veg=True, veg_a=0.667,
                     glcm_con=True, glcm_cor=True, glcm_ent=True,
                     glcm_by_x=1, glcm_by_y=1, glcm_radius=5, glcm_verbose=False,
                     conv_gauss_stdev=4, conv_method='nearest') -> _Frame2DChannel:
        """ Gets all implemented channels, excluding possible.

        Order is given by the argument order.
        R, G, B, H, S, V, EX_G, MEX_G, EX_GR, NDI, VEG, X, Y,
        ConR, ConG, ConB, CorrR, CorrG, CorrB, EntR, EntG, EntB

        :param self_: Include current frame
        :param xy: XY Coordinates
        :param hsv: Hue, Saturation, Value
        :param ex_g: Excess Green
        :param mex_g: Modified Excess Green
        :param ex_gr: Excess Green Minus Red
        :param ndi: Normalized Difference Index
        :param veg: Vegetative Index
        :param veg_a: Vegatative Index Const A
        :param glcm_con: GLCM Contrast
        :param glcm_cor: GLCM Correlation
        :param glcm_ent: GLCM Entropy
        :param glcm_by_x: GLCM By X Parameter
        :param glcm_by_y: GLCM By Y Parameter
        :param glcm_radius: GLCM Radius
        :param glcm_verbose: Whether to have glcm entropy generation give feedback
        :param conv_gauss_stdev: The stdev of the gaussian kernel used to convolute existing channels if GLCM is used
        :param conv_method: Can be 'average' or 'nearest'
        """

        features = \
            [self.data if self_ else None,
             self.get_xy() if xy else None,
             self.get_hsv() if hsv else None,
             self.get_ex_g() if ex_g else None,
             self.get_ex_g(True) if mex_g else None,
             self.get_ex_gr() if ex_gr else None,
             self.get_ndi() if ndi else None,
             self.get_veg(veg_a) if veg else None]

        frame = self.init(np.concatenate([f for f in features if f is not None], axis=2))

        if glcm_con or glcm_cor or glcm_ent:
            glcm = self.get_glcm(
                contrast=glcm_con, correlation=glcm_cor, entropy=glcm_ent,
                by_x=glcm_by_x, by_y=glcm_by_y, radius=glcm_radius, verbose=glcm_verbose
                )

            if frame:
                # We trim the frame so that the new glcm can fit
                # We also average the shifted frame with the current
                kernel_diam = glcm_radius * 2 + 1
                if conv_method == 'nearest':
                    kernel = np.zeros([kernel_diam + 1, kernel_diam + 1, 1])
                    kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] = 1
                else: # 'average'
                    kernel = np.outer(gaussian(kernel_diam + glcm_by_y, conv_gauss_stdev),
                                      gaussian(kernel_diam + glcm_by_x, conv_gauss_stdev))
                    kernel = np.expand_dims(kernel, axis=-1)
                fft = fftconvolve(frame.data, kernel, mode='valid', axes=[0,1])
                glcm = np.concatenate([fft, glcm], axis=2)
            return self.init(glcm)

        return frame
