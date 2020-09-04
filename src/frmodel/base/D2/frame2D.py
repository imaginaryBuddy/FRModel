from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import warnings
import pandas as pd
from skimage.color import rgb2hsv
import seaborn as sns
from PIL import Image

from frmodel.base.D2.channel2D import Channel2D
from frmodel.base.consts import CONSTS
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.preprocessing import minmax_scale as sk_minmax_scale
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from math import ceil

CHANNEL = CONSTS.CHANNEL

MAX_RGB = 255

@dataclass
class Frame2D:
    """ A Frame is an alias to an Image.

    The underlying representation is a 2D array, each cells holds a structured array, consisting of RGB Channels.

    It's easily extendable for more channels, but for now we only have the RGB.
    """

    data: np.ndarray

    class SplitMethod:
        DROP = 0
        CROP = 1

        # Require to support padding? Not implemented yet.

    def split_xy(self,
                 by: int,
                 method: SplitMethod = SplitMethod.DROP):
        """ Short hand for splitting by both axes.

        Splits by X first, then Y.

        E.g.::

            | 1 | 6  | 11 | 16 |
            | 2 | 7  | 12 | 17 |
            | 3 | 8  | 13 | 18 |
            | 4 | 9  | 14 | 19 |
            | 5 | 10 | 15 | 20 |

            [[1,2,3,4,5],[6,7,8,...], ...]

        """
        return [f.split(by, axis=CONSTS.AXIS.X, method=method)
                for f in self.split(by, axis=CONSTS.AXIS.Y, method=method)]

    def split(self,
              by: int,
              axis: CONSTS.AXIS = CONSTS.AXIS.Y,
              method: SplitMethod = SplitMethod.DROP) -> List[Frame2D]:
        """ Splits the current Frame into windows of specified size.

        This operation is much faster but cannot have overlapping windows

        E.g.::

            frame.split(
                by = 50,
                axis = CONSTS.AXIS.Y,
                method = Frame2D.SplitMethod.DROP
            )

        Will slice the images vertically, drops the last slice if it's not perfectly divisible.
        """

        # Pre-process by as modified by_
        # np.split_array splits it by the number of slices generated,
        # we need to transform this into the slice locations
        if axis == CONSTS.AXIS.Y:
            by_ = np.arange(by, self.width(), by)
        elif axis == CONSTS.AXIS.X:
            by_ = np.arange(by, self.height(), by)
        else:
            raise TypeError(f"Axis {axis} is not recognised. Use CONSTS.AXIS class.")

        spl = np.array_split(self.data, by_, axis=axis)
        if method == Frame2D.SplitMethod.CROP:
            # By default, it'll just "crop" the edges
            return [Frame2D(s) for s in spl]
        elif method == Frame2D.SplitMethod.DROP:
            # We will use a conditional to drop any images that is "cropped"
            return [Frame2D(s) for s in spl if s.shape[axis] == by]

    def slide_xy(self, by, stride=1):
        """ Short hand for sliding by both axes.

        Slides by X first, then Y.

        E.g.::

            | 1 | 6  | 11 | 16 |
            | 2 | 7  | 12 | 17 |
            | 3 | 8  | 13 | 18 |
            | 4 | 9  | 14 | 19 |
            | 5 | 10 | 15 | 20 |

            [[1,2,3,4,5],[6,7,8,...], ...]

        """
        return [f.slide(by=by, stride=stride, axis=CONSTS.AXIS.X)
                for f in self.slide(by=by, stride=stride, axis=CONSTS.AXIS.Y)]

    def slide(self, by, stride, axis=CONSTS.AXIS.Y):
        """ Slides a window along an axis and grabs that window as a new Frame2D

        This operation is slower due to looping but allows for overlapping windows

        If the window meets the edge, it will be dropped.

        :param by: The size of the window to slide
        :param stride: The stride of the window on specified axis
        :param axis: Axis to slide on
        :return:
        """

        if axis == CONSTS.AXIS.Y:
            return [Frame2D(self.data[:, i: i + by])
                    for i in range(0, self.width() - by + 1, stride)]
        elif axis == CONSTS.AXIS.X:
            return [Frame2D(self.data[i: i + by, :])
                    for i in range(0, self.height() - by + 1, stride)]

    @staticmethod
    def from_image(file_path: str) -> Frame2D:
        """ Creates an instance using the file path. """
        img = Image.open(file_path)
        ar = np.asarray(img)
        return Frame2D(ar)

    @staticmethod
    def from_rgbxy_(ar: np.ndarray, xy_pos=(3,4), width=None, height=None) -> Frame2D:
        """ Rebuilds the frame with XY values. XY should be of integer values, otherwise, will be casted.

        Note that RGB channels MUST be on index 0, 1, 2 else some functions may break. However, can be ignored.

        The frame will be rebuild and all data will be retained, including XY.

        :param ar: The array to rebuild
        :param xy_pos: The positions of X and Y.
        :param height: Height of expected image, if None, Max will be used
        :param width: Width of expected image, if None, Max will be used
        """
        max_y = height if height else np.max(ar[:,xy_pos[1]]) + 1
        max_x = width if width else np.max(ar[:,xy_pos[0]]) + 1

        fill = np.zeros(( ceil(max_y), ceil(max_x), ar.shape[-1]), dtype=ar.dtype)

        # Vectorized X, Y <- RGBXY... Assignment
        fill[ar[:, xy_pos[1]].astype(int),
             ar[:, xy_pos[0]].astype(int)] = ar[:]

        return Frame2D(fill)

    def get_hsv(self) -> np.ndarray:
        """ Creates a HSV Array """
        return rgb2hsv(self.data_rgb())

    def get_ex_g(self, modified=False) -> np.ndarray:
        """ Calculates the excessive green index

        Original: 2g - 1r - 1b
        Modified: 1.262G - 0.884r - 0.331b

        :param modified: Whether to use the modified or not. Refer to docstring
        """
        
        if modified:
            return 1.262 * self.data_idx(CHANNEL.RED) -\
                   0.884 * self.data_idx(CHANNEL.GREEN) -\
                   0.331 * self.data_idx(CHANNEL.BLUE)

        else:
            return 2 * self.data_idx(CHANNEL.RED) -\
                   self.data_idx(CHANNEL.GREEN) -\
                   self.data_idx(CHANNEL.BLUE)

    def get_ex_gr(self) -> np.ndarray:
        """ Calculates the excessive green minus excess red index

        2g - r - b - 1.4r + g = 3g - 2.4r - b
        """
        
        return 3 * self.data_idx(CHANNEL.RED) - 2.4 * self.data_idx(CHANNEL.GREEN) - self.data_idx(CHANNEL.BLUE)

    def get_ndi(self):
        """ Calculates the Normalized Difference Index

        g - r / (g + r)
        """

        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.nan_to_num(
                np.true_divide(self.data_idx(CHANNEL.GREEN).astype(np.int) -
                               self.data_idx(CHANNEL.RED).astype(np.int),
                               self.data_idx(CHANNEL.GREEN).astype(np.int) +
                               self.data_idx(CHANNEL.RED).astype(np.int)),
                copy=False, nan=0, neginf=0, posinf=0)

        return x
    
    def get_veg(self, const_a: float = 0.667):
        """ Calculates the Vegetative Index

        (g) / r ^ (a) * b ^ (1 - a)

        :param const_a: Constant A depends on the camera used.
        """

        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.nan_to_num(self.data_idx(CHANNEL.GREEN).astype(np.float) /
                              (np.power(self.data_idx(CHANNEL.RED).astype(np.float), const_a) *
                               np.power(self.data_idx(CHANNEL.BLUE).astype(np.float), 1 - const_a)),
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

    def get_idxs(self, self_=False, xy=False, hsv=False, ex_g=False,
                 mex_g=False, ex_gr=False, ndi=False, veg=False,
                 veg_a=0.667, glcm=False, glcm_by_x=1, glcm_by_y=1,
                 glcm_radius=5, glcm_verbose=False) -> Frame2D:
        return self.get_all_idxs(self_,xy,hsv,ex_g,mex_g,ex_gr,ndi,
                                 veg,veg_a,glcm,glcm_by_x,glcm_by_y,glcm_radius,glcm_verbose)

    def get_all_idxs(self, self_=True, xy=True, hsv=True, ex_g=True, mex_g=True,
                     ex_gr=True, ndi=True, veg=True, veg_a=0.667, glcm=True,
                     glcm_by_x=1, glcm_by_y=1, glcm_radius=5, glcm_verbose=False) -> Frame2D:
        """ Gets all implemented features.

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
            [self.data           if self_ else None,
             self.get_xy()       if xy else None,
             self.get_hsv()      if hsv else None,
             self.get_ex_g()     if ex_g else None,
             self.get_ex_g(True) if mex_g else None,
             self.get_ex_gr()    if ex_gr else None,
             self.get_ndi()      if ndi else None,
             self.get_veg(veg_a) if veg else None]

        frame = Frame2D(np.concatenate([f for f in features if f is not None], axis=2))

        if glcm:
            # We trim the frame so that the new glcm can fit
            # We also average the shifted frame with the current
            frame.data = (frame.data[glcm_by_y:,glcm_by_x:] + frame.data[:-glcm_by_y,:-glcm_by_x]) / 2
            frame.data = frame.data[glcm_radius:-glcm_radius, glcm_radius: -glcm_radius]

            return Frame2D(np.concatenate([
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

        frames_a = Frame2D(self.data_rgb()[:-by_y, :-by_x].astype(np.int32)).slide_xy(by=glcm_window, stride=1)
        frames_b = Frame2D(self.data_rgb()[by_y:, by_x:].astype(np.int32)).slide_xy(by=glcm_window, stride=1)
        out = np.zeros((self.height() - by_y - radius * 2,
                        self.width() - by_x - radius * 2,
                        3 * 3))  # RGB * Index count

        for col, (col_a, col_b) in enumerate(zip(frames_a, frames_b)):
            if verbose: print(f"Progress {100 * col / len(frames_b):.2f}% [{col} / {len(frames_b)}]")
            for row, (cell_a, cell_b) in enumerate(zip(col_a, col_b)):

                # Contrast
                contrast = np.sum((cell_a.data - cell_b.data) ** 2, axis=(0,1))

                # Correlation
                mean_x = np.mean(cell_a.data, axis=(0,1))
                mean_y = np.mean(cell_b.data, axis=(0,1))
                mean = mean_x - mean_y
                std_x = np.std(cell_a.data, axis=(0,1))
                std_y = np.std(cell_b.data, axis=(0,1))
                std = std_x * std_y

                with np.errstate(divide='ignore', invalid='ignore'):
                    correlation = np.sum(
                        np.nan_to_num(((cell_a.data * cell_b.data) - mean) / std,
                                      copy=False, nan=0, neginf=-1, posinf=1), axis=(0,1))

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
                                      for g in c.swapaxes(0,1)
                                      for i in np.unique(g, return_counts=True)[1::2]])

                out[row, col, :] = np.concatenate([contrast, correlation, entropy])

        return out

    def kmeans(self,
               clusters,
               fit_indexes,
               sample_weight=None,
               verbose=False,
               scaler=sk_normalize,
               plot_figure=False,
               xy_indexes=(3,4),
               scatter_size=0.2
               ) -> KMeans:
        """ Fits a KMeans on current frame, on certain axes

        Example::

            frame_idxs = frame.get_idxs(xy=True,hsv=True,veg=True)
            frame_idxs.kmeans(clusters=5, fit_indexes=[2,3,4,5],
                              plot_figure=True,xy_indexes=(0,1),scatter_size=1)

        This will firstly get the coordinate, hsv and veg indexes, placing it in a df

        :param clusters: The number of clusters
        :param fit_indexes: The indexes to fit
        :param sample_weight: The sample weight for each record, if any. Can be None.
        :param verbose: Whether to print out the KMeans' verbose log
        :param scaler: The scaler to use, must be a callable(np.ndarray)
        :param plot_figure: Whether to plot a figure or not, using plt.gcf()
        :param xy_indexes: The indexes of XY. Must be present for plotting.
        :param scatter_size: Size of the marker on the scatter plot
        :return:
        """
        flat = self.data_flatten()
        frame_xy_trans = scaler(flat[:, fit_indexes])
        km = KMeans(n_clusters=clusters, verbose=verbose) \
            .fit(frame_xy_trans,
                 sample_weight=frame_xy_trans[:, sample_weight] if sample_weight else None)
        if plot_figure:
            df = pd.DataFrame(np.append(flat, km.labels_[...,np.newaxis],axis=-1))
            df.columns = [f"c{e}" for e, _ in enumerate(df.columns)]
            sns.lmplot(data=df,
                       x=f'c{xy_indexes[0]}',
                       y=f'c{xy_indexes[1]}',
                       hue=df.columns[-1],
                       fit_reg=False,
                       legend=True,
                       legend_out=True,
                       scatter_kws={"s": scatter_size})
        return km

    def data_kdtree(self, leaf_size=40, metric='minkowski', **kwargs) -> KDTree:
        """ Constructs a KDTree with current data.

        Uses sklearn.neighbours.KDTree API."""
        return KDTree(self.data_flatten(),
                      leaf_size=leaf_size,
                      metric=metric,
                      **kwargs)

    def normalize(self) -> Frame2D:
        shape = self.data.shape
        return Frame2D(sk_normalize(self.data.reshape([-1, shape[-1]]), axis=0).reshape(shape))

    def minmax_scale(self) -> Frame2D:
        shape = self.data.shape
        return Frame2D(sk_minmax_scale(self.data.reshape([-1, shape[-1]]), axis=0).reshape(shape))

    def save(self, file_path: str, **kwargs) -> None:
        """ Saves the current Frame file """
        Image.fromarray(self.data_rgb().astype(np.uint8)).save(file_path, **kwargs)

    def size(self) -> np.ndarray:
        """ Returns the number of pixels """
        return self.data.size

    def data_flatten(self) -> np.ndarray:
        return self.data.reshape([-1, self.shape[-1]])

    @property
    def shape(self) -> Tuple:
        return self.data.shape

    def data_idx(self, idx):
        return self.data[..., [idx]]
    def data_idxs(self, idxs):
        return self.data[..., idxs]
    def data_rgb(self):
        return self.data[..., [CHANNEL.RED, CHANNEL.GREEN, CHANNEL.BLUE]]

    def height(self):
        return self.shape[0]

    def width(self):
        return self.shape[1]

    def channel(self, channel: CONSTS.CHANNEL) -> Channel2D:
        """ Gets the channel of the Frame as Channel 2D. """
        warnings.warn("Channel2D will be deprecated soon, it's recommended to work within Frame2D "
                      "or slice using np.ndarray.")
        return Channel2D(self.data[..., channel])
