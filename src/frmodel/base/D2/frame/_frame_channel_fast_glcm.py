from __future__ import annotations

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

        scaled = self.scale_values(to_min=CONSTS.BOUNDS.MIN_RGB, to_max=CONSTS.BOUNDS.MAX_RGB - 1).astype(np.uint16)

        windows = (scaled.view_windows(glcm.radius * 2 + 1,
                                       glcm.radius * 2 + 1, glcm.by, glcm.by) //
                   (CONSTS.BOUNDS.MAX_RGB // glcm.bins)).astype(np.uint16)

        windows_a, windows_b = windows[:-glcm.by, :-glcm.by], windows[glcm.by:, glcm.by:]
        # Combination Window
        windows = windows_a + windows_b * glcm.bins

        channels = self.shape[-1]
        windows_h = windows.shape[0]
        windows_w = windows.shape[1]

        bin_combined_maximum = glcm.bins ** 2

        result = np.zeros((bin_combined_maximum,
                           windows_h,  # Windows Height
                           windows_w,  # Windows Width
                           channels), dtype=np.uint8)

        result = cy_fast_glcm(windows, result, True)

        glcm_ar = result.reshape([glcm.bins, glcm.bins,
                                  windows_h,  # Windows Height
                                  windows_w,  # Windows Width
                                  channels])

        # We get the lengths to preemptively create a GLCM np.ndarray
        con_len = self._get_chn_size(glcm.contrast)
        cor_len = self._get_chn_size(glcm.correlation)
        ent_len = self._get_chn_size(glcm.entropy)

        data = np.zeros(shape=
                        [windows_h,
                         windows_w,
                         con_len + cor_len + ent_len])

        labels = []

        i = 0

        if glcm.contrast:
            data[..., i:i + con_len] =\
                self._get_glcm_contrast(
                    windows_a[..., self._labels_to_ix(glcm.contrast)],
                    windows_b[..., self._labels_to_ix(glcm.contrast)],
                    glcm_ar  [..., self._labels_to_ix(glcm.contrast)])
            i += con_len
            labels.extend(CONSTS.CHN.GLCM.CON(list(self._util_flatten(glcm.contrast))))

        if glcm.correlation:
            data[..., i:i + cor_len] =\
                self._get_glcm_correlation(
                    windows_a[..., self._labels_to_ix(glcm.correlation)],
                    windows_b[..., self._labels_to_ix(glcm.correlation)],
                    glcm_ar  [..., self._labels_to_ix(glcm.correlation)])
            i += cor_len
            labels.extend(CONSTS.CHN.GLCM.COR(list(self._util_flatten(glcm.correlation))))

        if glcm.entropy:
            if self.data.min() < CONSTS.BOUNDS.MIN_RGB or \
               self.data.max() >= CONSTS.BOUNDS.MAX_RGB:
                raise Exception(f"Minimum and Maximum for Entropy must be "
                                f"[{CONSTS.BOUNDS.MIN_RGB}, {CONSTS.BOUNDS.MAX_RGB}), "
                                f"received [{self.data.min()}, {self.data.max()}]")

            data[..., i:i + ent_len] =\
                self._get_glcm_entropy(glcm_ar[..., self._labels_to_ix(glcm.entropy)])
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
    def _get_glcm_correlation(a: np.ndarray,
                              b: np.ndarray,
                              glcm: np.ndarray) -> np.ndarray:
        """ Correlation

        :param a: Windows A
        :param b: Offset Windows B
        :param glcm: The GLCM Probability
        """

        a_mean = np.mean(a)
        b_mean = np.mean(b)
        a_std  = np.std(a)
        b_std  = np.std(b)

        return (np.sum(a,axis=(2,3)) - a_mean * a.shape[2] * a.shape[3]) * \
               (np.sum(b,axis=(2,3)) - b_mean * b.shape[2] * b.shape[3]) / \
               (a_std * b_std * a.shape[2] * a.shape[2])

    @staticmethod
    def _get_glcm_entropy(glcm: np.ndarray) -> np.ndarray:
        """ Gets the entropy

        :param glcm: Offset ar A
        """

        glcm_p = glcm / glcm.sum()
        ent = glcm_p * (-np.log(glcm_p + np.finfo(float).eps))
        return np.sum(ent, axis=(0,1))
