from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Tuple, List, TYPE_CHECKING

import numpy as np
from scipy.signal import fftconvolve
from skimage.util.shape import view_as_windows
from tqdm import tqdm

from frmodel.base import CONSTS
from frmodel.base.D2.frame._cy_corr import cy_corr
from frmodel.base.D2.frame._cy_entropy import cy_entropy

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D

MAX_RGB = 255

class _Frame2DChannelGLCM(ABC):

    @dataclass
    class GLCM:
        """ This holds all GLCM parameters to pass into get_glcm

        Note that contrast, correlation, entropy takes arguments similarly to how get_chns work.

        e.g. contrast=[f.CHN.HSV]
        """
        by_x:        int = 1
        by_y:        int = 1
        radius:      int = 5
        contrast:    List[CONSTS.CHN] = field(default_factory=[])
        correlation: List[CONSTS.CHN] = field(default_factory=[])
        entropy:     List[CONSTS.CHN] = field(default_factory=[])
        verbose:     bool = True

    def get_glcm(self: 'Frame2D', glcm:GLCM) -> Tuple[np.ndarray, List[str]]:
        """ This will get the GLCM statistics for this window

        Details on how GLCM works is shown on the wiki.

        :param glcm: A GLCM Class, this class holds all parameters.
        """

        # We get the lengths to preemptively create a GLCM np.ndarray
        con_len = self._get_chn_size(glcm.contrast)
        cor_len = self._get_chn_size(glcm.correlation)
        ent_len = self._get_chn_size(glcm.entropy)

        data = np.zeros(shape=[*self.crop_glcm(glcm.radius).shape[0:2], con_len + cor_len + ent_len])

        def pair_ar(ar: np.ndarray):
            return ar[:-glcm.by_y, :-glcm.by_x], ar[glcm.by_y:, glcm.by_x:]

        labels = []

        i = 0

        if glcm.contrast:
            data[..., i:i + con_len] =\
                self._get_glcm_contrast_py(*pair_ar(self.data_chn(glcm.contrast).data),
                                           radius=glcm.radius)
            i += con_len
            labels.extend(CONSTS.CHN.GLCM.CON(list(self._util_flatten(glcm.contrast))))

        if glcm.correlation:
            data[..., i:i + cor_len] =\
                self._get_glcm_correlation_cy(*pair_ar(self.data_chn(glcm.contrast).data),
                                              radius=glcm.radius, verbose=glcm.verbose)
            i += cor_len
            labels.extend(CONSTS.CHN.GLCM.COR(list(self._util_flatten(glcm.correlation))))

        if glcm.entropy:
            if any([c not in CONSTS.CHN.RGB for c in glcm.entropy]):
                raise Exception("Note that for non-RGB Entropies, it will be wildly incorrect as it assumes a "
                                "(0-255) value boundary")

            data[..., i:i + ent_len] =\
                self._get_glcm_entropy_cy(*pair_ar(self.data_chn(glcm.contrast).data),
                                          radius=glcm.radius, verbose=glcm.verbose)
            labels.extend(CONSTS.CHN.GLCM.ENT(list(self._util_flatten(glcm.entropy))))

        return data, labels

    @staticmethod
    def _get_glcm_contrast_py(ar_a: np.ndarray,
                              ar_b: np.ndarray,
                              radius) -> np.ndarray:
        """ Pure python implementation of Contrast.

        Cython isn't needed as it's purely vectorizable.

        Create the difference matrix, then convolve with a 1 filled kernel

        :param ar_a: Offset ar A
        :param ar_b: Offset ar B
        :param radius: Radius of window
        """

        ar = (ar_a - ar_b) ** 2
        return fftconvolve(ar, np.ones(shape=[radius * 2 + 1, radius * 2 + 1, 1]), mode='valid')

    @staticmethod
    def _get_glcm_correlation_py(ar_a: np.ndarray,
                                 ar_b: np.ndarray,
                                 radius) -> np.ndarray:
        """ Uses Pure Python to implement correlation GLCM

        :param ar_a: Offset ar A
        :param ar_b: Offset ar B
        :param radius: Radius of window
        """

        diam = radius * 2 + 1

        windows_a = view_as_windows(ar_a, [diam, diam, 3], step=1).squeeze()
        windows_a_mean = np.mean(windows_a, axis=(2, 3))
        windows_a_delta = windows_a - np.tile(np.expand_dims(windows_a_mean, (2, 3)), (1, 1, diam, diam, 1))
        windows_a_std = np.std(windows_a, axis=(2, 3))

        windows_b = view_as_windows(ar_b, [diam, diam, 3], step=1).squeeze()
        windows_b_mean = np.mean(windows_b, axis=(2, 3))
        windows_b_delta = windows_b - np.tile(np.expand_dims(windows_b_mean, (2, 3)), (1, 1, diam, diam, 1))
        windows_b_std = np.std(windows_b, axis=(2, 3))

        return np.divide(
            np.mean(windows_a_delta * windows_b_delta, axis=(2,3)),
            windows_a_std * windows_b_std,
            where=windows_a_std * windows_b_std != 0,
            out=np.zeros_like(windows_a_std)
        )

    @staticmethod
    def _get_glcm_correlation_naive_py(ar_a: np.ndarray,
                                       ar_b: np.ndarray,
                                       radius) -> np.ndarray:
        """ Legacy Naive Correlation Calculation in Pure Python

        This is naive because the formula doesn't seem to match up with the basis definition of Correlation.
        This omits the

        Using the following identity, we can vectorise it entirely!

        Var = E(X^2) - E(X)^2

        Corr = (a * b - (E(a) - E(b))) / std(a) * std(b)

        :param ar_a: Offset ar A
        :param ar_b: Offset ar B
        :param radius: Radius of window
        """

        kernel = np.ones(shape=[radius * 2 + 1, radius * 2 + 1, 1])

        conv_ab = fftconvolve(ar_a * ar_b, kernel, mode='valid')

        conv_a = fftconvolve(ar_a, kernel, mode='valid')
        conv_b = fftconvolve(ar_b, kernel, mode='valid')

        # E(A) & E(B)
        conv_ae = conv_a / kernel.size
        conv_be = conv_b / kernel.size

        conv_a2 = fftconvolve(ar_a ** 2, kernel, mode='valid')
        conv_b2 = fftconvolve(ar_b ** 2, kernel, mode='valid')

        # E(A^2) & E(B^2)
        conv_ae2 = conv_a2 / kernel.size
        conv_be2 = conv_b2 / kernel.size

        # E(A)^2 & E(B)^2
        conv_ae_2 = conv_ae ** 2
        conv_be_2 = conv_be ** 2

        conv_stda = np.sqrt(np.abs(conv_ae2 - conv_ae_2))
        conv_stdb = np.sqrt(np.abs(conv_be2 - conv_be_2))
        #
        # cor = (conv_ab - (conv_ae - conv_be) * (2 * radius + 1) ** 2) / \
        #       np.sqrt(conv_stda ** 2 * conv_stdb ** 2)

        with np.errstate(divide='ignore', invalid='ignore'):
            cor = (conv_ab - (conv_ae - conv_be) * (2 * radius + 1) ** 2) /\
                   np.sqrt(conv_stda ** 2 * conv_stdb ** 2)
            return np.nan_to_num(cor, copy=False, nan=0, neginf=-1, posinf=1)

    @staticmethod
    def _get_glcm_correlation_cy(ar_a: np.ndarray,
                                 ar_b: np.ndarray,
                                 radius: int,
                                 verbose: bool) -> Tuple[np.ndarray, Tuple[str]]:
        """ Correlation In Cython

        Uses the basis GLCM definition.

        :param ar_a: Offset ar A
        :param ar_b: Offset ar B
        :param radius: Radius of window
        """

        return cy_corr(ar_a.astype(np.uint8),
                       ar_b.astype(np.uint8),
                       radius, verbose) / ((2 * radius + 1) ** 2)

    @staticmethod
    def _get_glcm_entropy_cy(ar_a: np.ndarray,
                             ar_b: np.ndarray,
                             radius,
                             verbose
                             ) -> np.ndarray:
        """ Gets the entropy, uses the Cython entropy algorithm

        :param ar_a: Offset ar A
        :param ar_b: Offset ar B
        :param radius: Radius of window
        """
        rgb_c = ar_a + ar_b * (MAX_RGB + 1)

        return cy_entropy(rgb_c.astype(np.uint16), radius, verbose)

    @staticmethod
    def _get_glcm_entropy_py(ar_a: np.ndarray,
                             ar_b: np.ndarray,
                             radius,
                             verbose
                             ) -> np.ndarray:
        """ Gets the entropy in Pure Python

        :param ar_a: Offset ar A
        :param ar_b: Offset ar B
        :param radius: Radius of window
        :param verbose: Whether to show progress of entropy
        """
        # We make c = a + b * 256 (Notice 256 is the max val of RGB + 1).
        # The reason for this is so that we can represent (x, y) as a singular unique value.
        # This is a 1 to 1 mapping from a + b -> c, so c -> a + b is possible.
        # However, the main reason is that so that we can use np.unique without constructing
        # a tuple hash for each pair!

        ar_a = ar_a.astype(np.uint8)
        ar_b = ar_b.astype(np.uint8)

        cells = view_as_windows(ar_a * (MAX_RGB + 1) + ar_b,
                                [radius * 2 + 1, radius * 2 + 1, 3], step=1).squeeze()

        out = np.zeros((ar_a.shape[0] - radius * 2,
                        ar_a.shape[1] - radius * 2,
                        3))  # RGB count

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
