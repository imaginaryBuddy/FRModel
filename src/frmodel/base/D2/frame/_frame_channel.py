from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from skimage.color import rgb2hsv

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
                 veg_a=0.667, glcm=False, glcm_by_x=1, glcm_by_y=1,
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
        :param glcm: GLCM
        :param glcm_by_x: GLCM By X Parameter
        :param glcm_by_y: GLCM By Y Parameter
        :param glcm_radius: GLCM Radius
        :param glcm_verbose: Whether to have glcm generation give feedback
        """
        return self.get_all_chns(self_, xy, hsv, ex_g, mex_g, ex_gr, ndi,
                                 veg, veg_a, glcm, glcm_by_x, glcm_by_y, glcm_radius, glcm_verbose)

    def get_all_chns(self, self_=True, xy=True, hsv=True, ex_g=True, mex_g=True,
                     ex_gr=True, ndi=True, veg=True, veg_a=0.667, glcm=True,
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
        :param glcm: GLCM
        :param glcm_by_x: GLCM By X Parameter
        :param glcm_by_y: GLCM By Y Parameter
        :param glcm_radius: GLCM Radius
        :param glcm_verbose: Whether to have glcm generation give feedback
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

        if glcm:
            # We trim the frame so that the new glcm can fit
            # We also average the shifted frame with the current
            frame.data = (frame.data[glcm_by_y:, glcm_by_x:] + frame.data[:-glcm_by_y, :-glcm_by_x]) / 2
            frame.data = frame.data[glcm_radius:-glcm_radius, glcm_radius: -glcm_radius]

            return self.init(np.concatenate([
                frame.data,
                self.get_glcm(by_x=glcm_by_x, by_y=glcm_by_y,
                              radius=glcm_radius, verbose=glcm_verbose)], axis=2))

        return frame

    def get_glcm(self,
                 by_x: int = 1,
                 by_y: int = 1,
                 radius: int = 5,
                 verbose: bool = False):
        """ This will get the GLCM statistics for this window

        In order:

        Contrast_R, Contrast_G, Contrast_B,
        Correlation_R, Correlation_G, Correlation_B
        Entropy_R, Entropy_G, Entropy_B

        Note that the larger the GLCM stride, the more edge pixels will be removed.

        There will be edge cropping here, so take note of the following:

        1) Edge will be cropped on GLCM Making (That is shifting the frame with by_x and by_y.
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

        glcm_window = radius * 2 + 1

        frames_a = self.init(self.data_rgb()[:-by_y, :-by_x].astype(np.int32))\
            .slide_xy(by=glcm_window, stride=1)
        frames_b = self.init(self.data_rgb()[by_y:, by_x:].astype(np.int32))\
            .slide_xy(by=glcm_window, stride=1)
        out = np.zeros((self.height() - by_y - radius * 2,
                        self.width() - by_x - radius * 2,
                        3 * 3))  # RGB * Channel count

        for col, (col_a, col_b) in enumerate(zip(frames_a, frames_b)):
            if verbose: print(f"Progress {100 * col / len(frames_b):.2f}% [{col} / {len(frames_b)}]")
            for row, (cell_a, cell_b) in enumerate(zip(col_a, col_b)):
                # Contrast
                contrast = np.sum((cell_a.data - cell_b.data) ** 2, axis=(0, 1))

                # Correlation
                mean_x = np.mean(cell_a.data, axis=(0, 1))
                mean_y = np.mean(cell_b.data, axis=(0, 1))
                mean = mean_x - mean_y
                std_x = np.std(cell_a.data, axis=(0, 1))
                std_y = np.std(cell_b.data, axis=(0, 1))
                std = std_x * std_y

                with np.errstate(divide='ignore', invalid='ignore'):
                    correlation = np.sum(
                        np.nan_to_num(((cell_a.data * cell_b.data) - mean) / std,
                                      copy=False, nan=0, neginf=-1, posinf=1), axis=(0, 1))

                """ Optimized Entropy Calculation

                This is an abnormal and complicated way to optimize.

                1) We create c, which is the shape of a or b (they must be the same shape anyways)
                2) We make c = a + b * 256 (Notice 256 is the max val of RGB + 1).
                   The reason for this is so that we can represent (x, y) as a singular unique value.
                   This is a 1 to 1 mapping from a + b -> c, so c -> a + b is possible.
                   However, the main reason is that so that we can use np.unique without constructing
                   a tuple hash for each pair!
                3) After this, we just reshape with [-1, Channels]
                """
                c = np.zeros(cell_a.shape)
                c[:] = cell_a.data + cell_b.data * (MAX_RGB + 1)
                c = c.reshape([-1, c.shape[-1]])

                """ Entropy is complicated.

                Problem with unique is that it cannot unique on a certain axis as expected here,
                it's because of staggering dimension size, so we have to loop with a list comp.

                We swap axis because we want to loop on the channel instead of the c value.

                We call unique and grab the 2nd, 4th, ...th element because unique returns 2
                values here. The 2nd ones are the counts.

                Then we sum it up with np.sum, note that python sum is much slower on numpy arrays!
                """
                entropy = np.asarray([np.sum(i ** 2)
                                      for g in c.swapaxes(0, 1)
                                      for i in np.unique(g, return_counts=True)[1::2]])

                out[row, col, :] = np.concatenate([contrast, correlation, entropy])

        return out
