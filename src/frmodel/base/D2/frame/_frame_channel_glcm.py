from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import fftconvolve
from multiprocessing import Pool
from tqdm import tqdm

from skimage.util.shape import view_as_windows

from frmodel.base.consts import CONSTS

CHANNEL = CONSTS.CHANNEL
MAX_RGB = 255

class _Frame2DChannelGLCM(ABC):

    data: np.ndarray
    
    @abstractmethod
    def data_rgb(self): ...

    # noinspection PyArgumentList
    @classmethod
    def init(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def get_glcm(self,
                 by_x: int = 1,
                 by_y: int = 1,
                 radius: int = 5,
                 contrast: bool = True,
                 correlation: bool = True,
                 entropy: bool = True,
                 verbose: bool = False,
                 entropy_mp: bool = False,
                 entropy_mp_procs: int = None):
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

        idxs = [self._get_glcm_contrast(rgb_a, rgb_b, radius)                  if contrast else None,
                self._get_glcm_correlation(rgb_a, rgb_b, radius)               if correlation else None,
                self._get_glcm_entropy(rgb_a, rgb_b, radius, verbose,
                                       multiprocessing=entropy_mp,
                                       multiprocessing_procs=entropy_mp_procs) if entropy else None]

        # We drop the nones using a list comp
        return np.concatenate([i for i in idxs if i is not None], axis=2)

    def _get_glcm_contrast(self,
                           rgb_a: np.ndarray,
                           rgb_b: np.ndarray,
                           radius) -> np.ndarray:
        """ This is a faster implementation for contrast calculation.

        Create the difference matrix, then convolve with a 1 filled kernel

        :param rgb_a: Offset ar A
        :param rgb_b: Offset ar B
        :param radius: Radius of window
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

        :param rgb_a: Offset ar A
        :param rgb_b: Offset ar B
        :param radius: Radius of window
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
                          verbose,
                          multiprocessing,
                          multiprocessing_procs=None,
                          ) -> np.ndarray:
        """ Gets the entropy

        :param rgb_a: Offset ar A
        :param rgb_b: Offset ar B
        :param radius: Radius of window
        :param verbose: Whether to show progress of entropy
        :param multiprocessing: Whether to enable multiprocessing
        :param multiprocessing_procs: How many processes to run
        """
        # We make c = a + b * 256 (Notice 256 is the max val of RGB + 1).
        # The reason for this is so that we can represent (x, y) as a singular unique value.
        # This is a 1 to 1 mapping from a + b -> c, so c -> a + b is possible.
        # However, the main reason is that so that we can use np.unique without constructing
        # a tuple hash for each pair!

        rgb_a = rgb_a.astype(np.uint16)
        rgb_b = rgb_b.astype(np.uint16)

        cells = view_as_windows(self.init(rgb_a * (MAX_RGB + 1) + rgb_b).data,
                                [radius * 2 + 1, radius * 2 + 1,3], step=1).squeeze()

        out = np.zeros((rgb_a.shape[0] - radius * 2,
                        rgb_a.shape[1] - radius * 2,
                        3))  # RGB count

        # Branch to MP if True
        if multiprocessing:
            return self._get_glcm_entropy_mp(cells, out, verbose,
                                             procs=multiprocessing_procs)

        for row, _ in enumerate(tqdm(cells, total=len(cells), disable=not verbose)):
            for col, cell in enumerate(_):
                # We flatten the x and y axis first.
                c = cell.reshape([-1, cell.shape[-1]])

                """ Entropy is complicated.
    
                Problem with unique is that it cannot unique on a certain axis as expected here,
                it's because of staggering dimension size, so we have to loop with a list comp.
    
                We swap axis because we want to loop on the channel instead of the c value.
    
                We call unique and grab the 2nd, 4th, ...th element because unique returns 2
                values here. The 2nd ones are the counts.
    
                Then we sum it up with np.sum, note that python sum is much slower on numpy arrays!
                """

                entropy = np.asarray([np.sum(np.bincount(g) ** 2) for g in c.swapaxes(0, 1)])

                out[row, col, :] = entropy

        return out

    def _get_glcm_entropy_mp(self, cells, out, verbose, procs=None) -> np.ndarray:
        """ Branched from glcm_entropy, for Multiprocessing

        :param cells: Defined in glcm_entropy
        :param out: The array to throw values into, defined in glcm_entropy
        :param verbose: Whether to output progress
        :param procs: Number of processes to run in the multiprocessing
        """
        p = Pool(procs) if procs else Pool()

        for i, _ in enumerate(tqdm(p.imap_unordered(self._get_glcm_entropy_mp_loop, cells),
                                   total=len(cells), disable=not verbose)):
            out[i, :, :] = _

        return out

    @staticmethod
    def _get_glcm_entropy_mp_loop(row):
        """ This is a top-level method for pickling in multiprocessing """
        out = np.zeros(shape=[row.shape[0], row.shape[-1]])

        for i, cell in enumerate(row):
            # We flatten the x and y axis first.
            c = cell.reshape([-1, cell.shape[-1]])

            """ Entropy is complicated.

            Problem with unique is that it cannot unique on a certain axis as expected here,
            it's because of staggering dimension size, so we have to loop with a list comp.

            We swap axis because we want to loop on the channel instead of the c value.

            We call bincount to get call occurrences and square them.

            Then we sum it up with np.sum, note that python sum is much slower on numpy arrays!
            """

            out[i, :] = [np.sum(np.bincount(g) ** 2) for g in c.swapaxes(0, 1)]

        return out

    """ COO Method, (deprecated)
        def _get_glcm_entropy2(self,
                              rgb_a: np.ndarray,
                              rgb_b: np.ndarray,
                              radius,
                              verbose) -> np.ndarray:
            "Uses the COO Matrix to calculate Entropy, slightly slower."
    
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
    """