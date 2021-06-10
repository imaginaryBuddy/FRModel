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

        scale_on_bands: bool = False

        channels:     List[CONSTS.CHN] = field(default_factory=lambda: [])
        channel_maxs: List[float]      = field(default_factory=lambda: [])
        channel_mins: List[float]      = field(default_factory=lambda: [])
        verbose:     bool = True

    def get_glcm(self: 'Frame2D', glcm:GLCM) -> Tuple[np.ndarray, List[str]]:
        """ This will get the GLCM statistics for this window

        Details on how GLCM works is shown on the wiki.

        :param glcm: A GLCM Class, this class holds all parameters.
        """

        N_FEATURES = 5
        if glcm.bins <= 0 or (glcm.bins & glcm.bins - 1) != 0:
            raise Exception("glcm.bins must be a power of 2.")

        if glcm.scale_on_bands:
            scaled = self.scale_values_on_band(to_min=0, to_max=glcm.bins - 1).astype(np.uint8)
            windows = scaled.view_windows(glcm.radius * 2 + 1, glcm.radius * 2 + 1,
                                          glcm.by, glcm.by)
        else:
            scaled = self.scale_values_independent(
                from_min=glcm.channel_mins if glcm.channel_mins else None,
                from_max=glcm.channel_maxs if glcm.channel_maxs else None,
                to_min=CONSTS.BOUNDS.MIN_RGB,
                to_max=CONSTS.BOUNDS.MAX_RGB - 1).astype(np.uint8)

            windows = (scaled.view_windows(glcm.radius * 2 + 1,
                                           glcm.radius * 2 + 1, glcm.by, glcm.by) //
                       (CONSTS.BOUNDS.MAX_RGB // glcm.bins)).astype(np.uint8)


        windows_a, windows_b = windows[:-glcm.by, :-glcm.by], windows[glcm.by:, glcm.by:]
        # Combination Window
        windows_h = windows_a.shape[0]
        windows_w = windows_a.shape[1]

        # FAST GLCM
        result = cy_fast_glcm(windows_a, windows_b, True)
        n_chns = len(list(self._util_flatten(glcm.channels)))

        # We get the lengths to preemptively create a GLCM np.ndarray
        data = np.zeros(shape=[windows_h, windows_w, n_chns * N_FEATURES], dtype=np.float)

        labels = []

        data[..., 0: n_chns]         = result[0]
        data[..., n_chns:n_chns*2]   = result[1]
        data[..., n_chns*2:n_chns*3] = result[2]
        data[..., n_chns*3:n_chns*4] = result[3]
        data[..., n_chns*4:n_chns*5] = result[4]

        labels.extend(CONSTS.CHN.GLCM.CON( list(self._util_flatten(glcm.channels))))
        labels.extend(CONSTS.CHN.GLCM.COR( list(self._util_flatten(glcm.channels))))
        labels.extend(CONSTS.CHN.GLCM.ASM( list(self._util_flatten(glcm.channels))))
        labels.extend(CONSTS.CHN.GLCM.MEAN(list(self._util_flatten(glcm.channels))))
        labels.extend(CONSTS.CHN.GLCM.VAR( list(self._util_flatten(glcm.channels))))

        return data, labels
