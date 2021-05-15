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

        Note that contrast, correlation, asm takes arguments similarly to how get_chns work.

        Entropy has been replaced by ASM. Angular Second Moment

        e.g. contrast=[f.CHN.HSV]
        """
        by:          int = 1
        radius:      int = 2
        bins:        int = 8
        contrast:    List[CONSTS.CHN] = field(default_factory=lambda: [])
        correlation: List[CONSTS.CHN] = field(default_factory=lambda: [])
        asm:         List[CONSTS.CHN] = field(default_factory=lambda: [])
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
        ent_len = self._get_chn_size(glcm.asm)

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

        if glcm.asm:
            data[..., i:i + ent_len] = result[2][..., self._labels_to_ix(glcm.asm)]
            labels.extend(CONSTS.CHN.GLCM.ENT(list(self._util_flatten(glcm.asm))))

        return data, labels
