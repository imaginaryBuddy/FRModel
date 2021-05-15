from __future__ import annotations

import copy
from abc import ABC
from dataclasses import dataclass, field
from typing import Tuple, List, TYPE_CHECKING

import numpy as np

from frmodel.base import CONSTS
from frmodel.base.D2.frame._cy_fast_glcm import cy_fast_glcm

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D

class _Frame2DChannelFastGLCM(ABC):
    """ This re-implements Wang Jifei's Fast GLCM Script by adding the option of binarization. """

    @dataclass
    class GLCM:
        """ This holds all GLCM parameters to pass into get_glcm

        Note that contrast, correlation, entropy takes arguments similarly to how get_chns work.

        e.g. contrast=[f.CHN.HSV]
        """
        by:          int = 1
        radius:      int = 2
        bins:        int = 8
        contrast:    List[CONSTS.CHN] = field(default_factory=lambda: [])
        correlation: List[CONSTS.CHN] = field(default_factory=lambda: [])
        entropy:     List[CONSTS.CHN] = field(default_factory=lambda: [])
        verbose:     bool = True

    def get_glcm(self: 'Frame2D', glcm:GLCM) -> Tuple[np.ndarray, List[str]]:
        """ This will get the GLCM statistics for this window

        Details on how GLCM works is shown on the wiki.

        :param glcm: A GLCM Class, this class holds all parameters.
        """

        if glcm.bins % 2 != 0:
            raise Exception("glcm.bins must be a multiple of 2.")

        scaled = self.scale_values(to_min=CONSTS.BOUNDS.MIN_RGB, to_max=CONSTS.BOUNDS.MAX_RGB - 1).astype(np.uint8)

        windows = (scaled.view_windows(glcm.radius * 2 + 1,
                                       glcm.radius * 2 + 1, glcm.by, glcm.by) //
                   (CONSTS.BOUNDS.MAX_RGB // glcm.bins)).astype(np.uint8)

        windows_a, windows_b = windows[:-glcm.by, :-glcm.by], windows[glcm.by:, glcm.by:]
        # Combination Window
        windows_h = windows_a.shape[0]
        windows_w = windows_a.shape[1]

        # FAST GLCM
        result = cy_fast_glcm(windows_a, windows_b, True)

        # We get the lengths to preemptively create a GLCM np.ndarray
        con_len = self._get_chn_size(glcm.contrast)
        cor_len = self._get_chn_size(glcm.correlation)
        ent_len = self._get_chn_size(glcm.entropy)

        data = np.zeros(shape=[windows_h, windows_w, con_len + cor_len + ent_len])

        labels = []

        i = 0

        if glcm.contrast:
            data[..., i:i + con_len] = result[0][..., self._labels_to_ix(glcm.contrast)]
            i += con_len
            labels.extend(CONSTS.CHN.GLCM.CON(list(self._util_flatten(glcm.contrast))))

        if glcm.correlation:
            data[..., i:i + cor_len] = result[1][..., self._labels_to_ix(glcm.correlation)]
            i += cor_len
            labels.extend(CONSTS.CHN.GLCM.COR(list(self._util_flatten(glcm.correlation))))

        if glcm.entropy:
            data[..., i:i + ent_len] = result[2][..., self._labels_to_ix(glcm.entropy)]
            labels.extend(CONSTS.CHN.GLCM.ENT(list(self._util_flatten(glcm.entropy))))

        return data, labels

    @staticmethod
    def _get_glcm_contrast(a: np.ndarray,
                           b: np.ndarray,
                           glcm: np.ndarray) -> np.ndarray:
        """ Pure python implementation of Contrast.

        Cython isn't needed as it's purely vectorizable.

        Create the difference matrix, then convolve with a 1 filled kernel

        :param a: Windows A
        :param b: Offset Windows B
        :param glcm: The GLCM
        """

        windows = (a - b) ** 2
        con = np.sum(windows, axis=(2,3))
        return con / glcm.sum()

    @staticmethod
    def _get_glcm_mean(a: np.ndarray,
                       b: np.ndarray) -> np.ndarray:

        a_mean = np.mean(a, axis=(2,3))
        b_mean = np.mean(b, axis=(2,3))
        return (a_mean + b_mean) // 2

    @staticmethod
    def _get_glcm_stdev(a: np.ndarray,
                        b: np.ndarray) -> np.ndarray:

        a_mean = np.mean(a, axis=(2,3))
        b_mean = np.mean(b, axis=(2,3))
        return a_mean + b_mean

    @staticmethod
    def _get_glcm_correlation(a: np.ndarray,
                              b: np.ndarray,
                              glcm: np.ndarray) -> np.ndarray:
        """ Correlation

        :param a: Windows A
        :param b: Offset Windows B
        :param glcm: The GLCM Probability
        """

        a_ = np.sum(a, axis=(2,3))
        b_ = np.sum(b, axis=(2,3))

        a_mean = np.mean(a, axis=(2,3), dtype=np.float16)
        b_mean = np.mean(b, axis=(2,3), dtype=np.float16)
        a_std  = np.std(a, axis=(2,3), dtype=np.float16)
        b_std  = np.std(b, axis=(2,3), dtype=np.float16)

        with np.errstate(all='ignore'):
            return np.where(a_std * b_std != 0, (a_ - a_mean) * (b_ * b_mean) / a_std * b_std, 0) * \
                   np.sum(glcm, axis=(0,1)) / np.sum(glcm)

    @staticmethod
    def _get_glcm_entropy(glcm: np.ndarray) -> np.ndarray:
        """ Gets the entropy

        :param glcm: Offset ar A
        """

        glcm_p = glcm / glcm.sum()
        ent = glcm_p * (-np.log(glcm_p + np.finfo(float).eps))
        return np.sum(ent, axis=(0,1))
