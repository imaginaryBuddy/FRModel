from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from skimage.color import rgb2hsv
from scipy.signal import fftconvolve
from scipy.signal.windows import gaussian

from frmodel.base.consts import CONSTS

CHANNEL = CONSTS.CHANNEL
MAX_RGB = 255

class _Frame2DChannel(ABC):

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
        Modified: 1.262G - 0.884r - 0.331b

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

        g - r / (g + r)
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

        (g) / r ^ (a) * b ^ (1 - a)

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
                 glcm_radius=5, glcm_verbose=False) -> _Frame2DChannel:
        """ Gets selected channels

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
        """
        return self.get_all_chns(self_, xy, hsv, ex_g, mex_g, ex_gr, ndi,
                                 veg, veg_a, glcm_con, glcm_cor, glcm_ent,
                                 glcm_by_x, glcm_by_y, glcm_radius, glcm_verbose)

    def get_all_chns(self, self_=True, xy=True, hsv=True, ex_g=True, mex_g=True,
                     ex_gr=True, ndi=True, veg=True, veg_a=0.667,
                     glcm_con=True, glcm_cor=True, glcm_ent=True,
                     glcm_by_x=1, glcm_by_y=1, glcm_radius=5, glcm_verbose=False) -> _Frame2DChannel:
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
            # We trim the frame so that the new glcm can fit
            # We also average the shifted frame with the current

            kernel_diam = glcm_radius * 2 + 1
            kernel = np.outer(gaussian(kernel_diam + glcm_by_y, 8),
                              gaussian(kernel_diam + glcm_by_x, 8))
            kernel = np.expand_dims(kernel, axis=-1)
            fft = fftconvolve(frame.data, kernel, mode='valid', axes=[0,1])
            glcm = self.get_glcm(
                contrast=glcm_con, correlation=glcm_cor, entropy=glcm_ent,
                by_x=glcm_by_x, by_y=glcm_by_y, radius=glcm_radius, verbose=glcm_verbose)

            return self.init(np.concatenate([fft, glcm], axis=2))

        return frame

    def get_glcm(self,
                 by_x: int = 1,
                 by_y: int = 1,
                 radius: int = 5,
                 contrast: bool = True,
                 correlation: bool = True,
                 entropy: bool = True,
                 verbose: bool = False):
        """ This will get the GLCM statistics for this window

        In order:

        Contrast_R, Contrast_G, Contrast_B,
        Correlation_R, Correlation_G, Correlation_B
        Entropy_R, Entropy_G, Entropy_B

        Note that the larger the GLCM stride, the more edge pixels will be removed.

        There will be edge cropping here, so take note of the following:

        1) Edge will be cropped on GLCM Making (That is shifting the frame with by_x and by_y).
        2) Edge will be further cropped by GLCM Neighbour convolution.

        If only a specific GLCM is needed, open up an issue on GitHub, I just don't think it's needed right now.

        Consider this::

            1) GLCM Making, by_x = 1, by_y = 1
            o o o o o       | o o o o |       <B>            | o o o o |
            o o o o o       | o o o o |   | o o o o |  func  | o o o o |
            o o o o o  -->  | o o o o | + | o o o o |  --->  | o o o o |
            o o o o o       | o o o o |   | o o o o |        | o o o o |
            o o o o o           <A>       | o o o o |            <C>

            2) GLCM Neighbour Summation, radius = 1
                              o: Centre, +: Neighbour
            | o o o o |       | + + + x | , | x + + + | , | x x x x | , | x x x x |
            | o o o o |       | + o + x | , | x + o + | , | x + + + | , | + + + x |
            | o o o o |  -->  | + + + x | , | x + + + | , | x + o + | , | + o + x |
            | o o o o |       | x x x x | , | x x x x | , | x + + + | , | + + + x |
                <C>
            x x x x x  =>                 Note that it's slightly off centre because of (1)
            x o o x x  =>  | o o |
            x o o x x  =>  | o o |
            x x x x x  =>
            x x x x x  =>
            Original       Transformed

            The resultant size, if by_x = by_y
            frame.size - by - radius * 2
        """

        rgb = self.data_rgb().astype(np.int32)
        rgb_a = rgb[:-by_y, :-by_x]
        rgb_b = rgb[by_y:, by_x:]

        idxs = [self._get_glcm_contrast(rgb_a, rgb_b, radius)            if contrast else None,
                self._get_glcm_correlation(rgb_a, rgb_b, radius)         if correlation else None,
                self._get_glcm_entropy(rgb_a, rgb_b, radius, verbose)    if entropy else None]

        # We drop the nones using a list comp
        return np.concatenate([i for i in idxs if i is not None], axis=2)

    def _get_glcm_contrast(self,
                           rgb_a: np.ndarray,
                           rgb_b: np.ndarray,
                           radius) -> np.ndarray:
        """ This is a faster implementation for contrast calculation.

        Create the difference matrix, then convolve with a 1 filled kernel
        """

        ar = (rgb_a - rgb_b) ** 2
        return fftconvolve(ar, np.ones(shape=[radius * 2 + 1, radius * 2 + 1, 1]), mode='valid')

    def _get_glcm_correlation(self,
                              rgb_a: np.ndarray,
                              rgb_b: np.ndarray,
                              radius) -> np.ndarray:
        """ This is a faster implementation for correlation calculation.

        Using the following identity, we can vectorise it entirely!

        Var = E(X^2) - E(X)^2

        Corr = (a * b - (E(a) - E(b))) / std(a) * std(b)
        """

        kernel = np.ones(shape=[radius * 2 + 1, radius * 2 + 1, 1])

        conv_ab = fftconvolve(rgb_a * rgb_b, kernel, mode='valid')

        conv_a = fftconvolve(rgb_a, kernel, mode='valid')
        conv_b = fftconvolve(rgb_b, kernel, mode='valid')

        # E(A) & E(B)
        conv_ae = conv_a / kernel.size
        conv_be = conv_b / kernel.size

        conv_a2 = fftconvolve(rgb_a ** 2, kernel, mode='valid')
        conv_b2 = fftconvolve(rgb_b ** 2, kernel, mode='valid')

        # E(A^2) & E(B^2)
        conv_ae2 = conv_a2 / kernel.size
        conv_be2 = conv_b2 / kernel.size

        # E(A)^2 & E(B)^2
        conv_ae_2 = conv_ae ** 2
        conv_be_2 = conv_be ** 2

        conv_stda = np.sqrt(np.abs(conv_ae2 - conv_ae_2))
        conv_stdb = np.sqrt(np.abs(conv_be2 - conv_be_2))

        with np.errstate(divide='ignore', invalid='ignore'):
            cor = (conv_ab - (conv_ae - conv_be)) / conv_stda * conv_stdb
            return np.nan_to_num(cor, copy=False, nan=0, neginf=-1, posinf=1)

    def _get_glcm_entropy(self,
                          rgb_a: np.ndarray,
                          rgb_b: np.ndarray,
                          radius,
                          verbose) -> np.ndarray:

        # We make c = a + b * 256 (Notice 256 is the max val of RGB + 1).
        # The reason for this is so that we can represent (x, y) as a singular unique value.
        # This is a 1 to 1 mapping from a + b -> c, so c -> a + b is possible.
        # However, the main reason is that so that we can use np.unique without constructing
        # a tuple hash for each pair!

        windows = self.init(rgb_a * (MAX_RGB + 1) + rgb_b)\
                      .slide_xy(by=radius * 2 + 1, stride=1)

        out = np.zeros((rgb_a.shape[0] - radius * 2,
                        rgb_a.shape[1] - radius * 2,
                        3))  # RGB * Channel count

        for col, _ in enumerate(windows):
            if verbose: print(f"GLCM Entropy: {col} / {len(windows)}")
            for row, cell in enumerate(_):
                # We flatten the x and y axis first.
                c = cell.data.reshape([-1, cell.shape[-1]])

                """ Entropy is complicated.

                Problem with unique is that it cannot unique on a certain axis as expected here,
                it's because of staggering dimension size, so we have to loop with a list comp.

                We swap axis because we want to loop on the channel instead of the c value.

                We call unique and grab the 2nd, 4th, ...th element because unique returns 2
                values here. The 2nd ones are the counts.

                Then we sum it up with np.sum, note that python sum is much slower on numpy arrays!
                """
                entropy = np.asarray([np.sum(i * np.log2(i))
                                      for g in c.swapaxes(0, 1)
                                      for i in np.unique(g, return_counts=True)[1::2]])

                out[row, col, :] = entropy

        return out

    def _get_glcm_entropy2(self,
                          rgb_a: np.ndarray,
                          rgb_b: np.ndarray,
                          radius,
                          verbose) -> np.ndarray:
        """ Uses the COO Matrix to calculate Entropy, slightly slower. """

        w_a = self.init(rgb_a).slide_xy(by=radius * 2 + 1, stride=1)
        w_b = self.init(rgb_b).slide_xy(by=radius * 2 + 1, stride=1)

        out = np.zeros((rgb_a.shape[0] - radius * 2,
                        rgb_a.shape[1] - radius * 2,
                        3))  # RGB * Channel count

        for col, (_a, _b) in enumerate(zip(w_a, w_b)):
            if verbose: print(f"GLCM Entropy: {col} / {len(w_a)}")
            for row, (ca, cb) in enumerate(zip(_a, _b)):
                # We flatten the x and y axis first.
                ca = ca.data.reshape([-1, ca.shape[-1]])
                cb = cb.data.reshape([-1, cb.shape[-1]])
                cd = np.ones(ca.shape[0])
                
                coo_r = coo_matrix((cd, (ca[..., 0], cb[..., 0])), shape=(MAX_RGB, MAX_RGB)).tocsr(copy=False).power(2).sum()
                coo_g = coo_matrix((cd, (ca[..., 1], cb[..., 1])), shape=(MAX_RGB, MAX_RGB)).tocsr(copy=False).power(2).sum()
                coo_b = coo_matrix((cd, (ca[..., 2], cb[..., 2])), shape=(MAX_RGB, MAX_RGB)).tocsr(copy=False).power(2).sum()

                out[row, col, :] = [coo_r, coo_g, coo_b]

        return out
