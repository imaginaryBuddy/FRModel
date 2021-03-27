from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Tuple, List, TYPE_CHECKING

import numpy as np
from scipy.signal import fftconvolve
from skimage.util.shape import view_as_windows
from sklearn.preprocessing import normalize
from tqdm import tqdm

from frmodel.base import CONSTS
from frmodel.base.D2.frame._cy_corr import cy_corr
from frmodel.base.D2.frame._cy_entropy import cy_entropy

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D


class _Frame2DChannelSpec(ABC):
    """ This is a separate class to handle spectral channels, so as to not clutter the main class too much"""

    def _e(self: 'Frame2D'):
        """ Short forms for easy calling, not recommended to use outside of class scope """
        return self.data_chn(self.CHN.RED_EDGE).data
    def _n(self: 'Frame2D'):
        """ Short forms for easy calling, not recommended to use outside of class scope """
        return self.data_chn(self.CHN.NIR).data

    def get_ndvi(self: 'Frame2D') -> np.ndarray:
        """ Normalized Difference Vegetation Index """
        return (self._n() - self._r()) / (self._n() + self._r())
    def get_bndvi(self: 'Frame2D') -> np.ndarray:
        """ Blue Normalized Difference Vegetation Index """
        return (self._n() - self._b()) / (self._n() + self._b())
    def get_gndvi(self: 'Frame2D') -> np.ndarray:
        """ Green Normalized Difference Vegetation Index """
        return (self._n() - self._g()) / (self._n() + self._g())
    def get_gari(self: 'Frame2D') -> np.ndarray:
        """ Green Atmospherically Resistant Vegetation Index """
        b_r = self._b() - self._r()
        return (self._n() - (self._g() - b_r)) / (self._n() - (self._g() + b_r))
    def get_gli(self: 'Frame2D') -> np.ndarray:
        """ Green Leaf Index """
        return (2 * self._g() - self._r() - self._b()) / (2 * self._g() + self._r() + self._b())
    def get_gbndvi(self: 'Frame2D') -> np.ndarray:
        """ Green Blue NDVI """
        return (self._n() - self._b()) / (self._n() + self._b())
    def get_grndvi(self: 'Frame2D') -> np.ndarray:
        """ Green Red NDVI """
        return (self._n() - self._g()) / (self._n() + self._g())
    def get_ndre(self: 'Frame2D') -> np.ndarray:
        """ Normalized Difference Red Edge """
        return (self._n() - self._e()) / (self._n() + self._e())
    def get_lci(self: 'Frame2D') -> np.ndarray:
        """ Leaf Chlorophyll Index  """
        return (self._n() - self._e()) / (self._n() + self._r())
    def get_msavi(self: 'Frame2D') -> np.ndarray:
        """ Modified Soil Adjusted Vegetation Index """
        aux = (2 * self._n() + 1)
        return (aux - np.sqrt(aux ** 2 - 8 * (self._n() - self._r()))) / 2
    def get_osavi(self: 'Frame2D') -> np.ndarray:
        """ Optimized Soil Adjusted Vegetation Index """
        return (self._n() - self._r()) / (self._n() + self._r() + 0.16)

"""
        RED_EDGE    = "RE"
        NIR         = "NIR"
        NDVI        = "NDVI"
        NDWI        = "NDWI"
        GNDVI       = "GNDVI"
        OSAVI       = "OSAVI"
        NDRE        = "NDRE"
        LCI         = "LCI"
"""
