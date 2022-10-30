from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Iterable

import numpy as np
from skimage.color import rgb2hsv

from frmodel.base.D2.frame._frame_channel_fast_glcm import _Frame2DChannelFastGLCM
# from frmodel.base.D2.frame._frame_channel_glcm import _Frame2DChannelGLCM
from frmodel.base.D2.frame._frame_channel_spec import _Frame2DChannelSpec

if TYPE_CHECKING:
    from frmodel.base.D2.frame2D import Frame2D


class _Frame2DChannel(_Frame2DChannelFastGLCM, _Frame2DChannelSpec):

    def _r(self: 'Frame2D'):
        """ Short forms for easy calling, not recommended to use outside of class scope """
        return self.data_chn(self.CHN.RED).data
    def _g(self: 'Frame2D'):
        """ Short forms for easy calling, not recommended to use outside of class scope """
        return self.data_chn(self.CHN.GREEN).data
    def _b(self: 'Frame2D'):
        """ Short forms for easy calling, not recommended to use outside of class scope """
        return self.data_chn(self.CHN.BLUE).data

    def get_all_chns(self: 'Frame2D',
                     self_: bool = False,
                     exc_chns: Iterable[Frame2D.CHN] = None,
                     glcm: _Frame2DChannel.GLCM = None) -> 'Frame2D':
        """ Gets all channels, excludes any in exc_chns.

        Use get_chns to get selected ones

        Order is given by the argument order.
        R, G, B, X, Y, H, S, V, EX_G, MEX_G, EX_GR, NDI, VEG,
        ConR, ConG, ConB, CorrR, CorrG, CorrB, EntR, EntG, EntB

        :param self_: Include current frame
        :param exc_chns: Excluded Channels
        :param glcm: GLCM Object

        :returns: Frame2D Object with requested channesls
        """

        self: 'Frame2D'
        chns = [self.CHN.XY,
                self.CHN.EX_G,
                self.CHN.MEX_G,
                self.CHN.HSV,
                self.CHN.NDI,
                self.CHN.VEG,
                self.CHN.EX_GR]

        return self.get_all_chns(self_, [c for c in chns if c not in exc_chns], glcm)

    def get_chns(self: 'Frame2D',
                 self_: bool = True,
                 chns: Iterable[Frame2D.CHN] = None,
                 glcm: _Frame2DChannel.GLCM = None) -> 'Frame2D':
        """ Gets selected channels

        Use get_all_chns to get all but selected ones

        Order is given by the argument order.
        R, G, B, H, S, V, EX_G, MEX_G, EX_GR, NDI, VEG, X, Y,
        ConR, ConG, ConB, CorrR, CorrG, CorrB, EntR, EntG, EntB

        :param self_: Include current frame
        :param chns: Channels
        :param glcm: GLCM Object

        :returns
        """

        labels = []
        chns = [] if not chns else chns

        self: 'Frame2D'
        data = np.zeros([*self.shape[0:2],
                          self._get_chn_size(chns) + self.shape[-1] if self_ else
                          self._get_chn_size(chns)])

        _chn_mapping: dict = {
            self.CHN.XY     : self.get_xy,
            self.CHN.HSV    : self.get_hsv,
            self.CHN.EX_G   : self.get_ex_g,
            self.CHN.EX_GR  : self.get_ex_gr,
            self.CHN.MEX_G  : self.get_mex_g,
            self.CHN.NDI    : self.get_ndi,
            self.CHN.VEG    : self.get_ex_g,

            self.CHN.NDVI   : self.get_ndvi,
            self.CHN.BNDVI  : self.get_bndvi,
            self.CHN.GNDVI  : self.get_gndvi,
            self.CHN.GARI   : self.get_gari,
            self.CHN.GLI    : self.get_gli,
            self.CHN.GBNDVI : self.get_gbndvi,
            self.CHN.GRNDVI : self.get_grndvi,
            self.CHN.NDRE   : self.get_ndre,
            self.CHN.LCI    : self.get_lci,
            self.CHN.MSAVI  : self.get_msavi,
            self.CHN.OSAVI  : self.get_osavi,
        }

        it = 0

        if self_:
            data[..., it:self.shape[-1]] = self.data
            labels = [self.labels.keys()]
            it += self.shape[-1]

        for chn in chns:
            length = len(chn) if isinstance(chn, Tuple) else 1
            try:
                try:
                    # The case where the channel is already calculated
                    data[..., it:it+length] = self.data_chn(chn).data
                except KeyError:
                    # The case where the channel is missing
                    data[..., it:it+length] = _chn_mapping[chn]()
                labels.append(chn)
            except KeyError:
                if chn in (self.CHN.X, self.CHN.Y):
                    raise KeyError(f"You cannot get {chn} separately from XY, call with CHN.HSV")
                elif chn in self.CHN.HSV:
                    raise KeyError(f"You cannot get {chn} separately from HSV, call with CHN.HSV")
                else:
                    raise KeyError(f"Failed to find channel {chn}, I recommend to use CONSTS.CHN to get the correct"
                                   f"strings to call")
            it += length

        frame: 'Frame2D' = self.create(data=data, labels=labels)

        if glcm:
            if frame.shape[-1] == 0:
                # Cannot convolute a 0 set. We'll still entertain get_glcm only.
                frame = self.create(*self.get_glcm(glcm))
            else:
                frame = frame.astype(np.float)
                frame = frame.crop_glcm(glcm.radius)\
                    .append(*self.get_glcm(glcm))

        frame._data = np.ma.array(frame.data,
                                  mask=np.repeat(frame.nanmask()[..., np.newaxis], frame.shape[-1]))
        return frame

    def get_hsv(self: 'Frame2D') -> np.ndarray:
        """ Creates a HSV

        :return: np.ndarray, similar shape to callee. Use get_chns to get as Frame2D
        """
        return rgb2hsv(self.data_rgb().data)

    def get_ex_g(self: 'Frame2D') -> np.ndarray:
        """ Calculates the excessive green index

        Original: 2g - 1r - 1b

        :return: np.ndarray, similar shape to callee. Use get_chns to get as Frame2D
        """

        return 2 * self.data_chn(self.CHN.RED).data - \
               self.data_chn(self.CHN.GREEN).data - \
               self.data_chn(self.CHN.BLUE).data

    def get_mex_g(self: 'Frame2D') -> np.ndarray:
        """ Calculates the Modified excessive green index

        1.262g - 0.884r - 0.331b

        :return: np.ndarray, similar shape to callee. Use get_chns to get as Frame2D
        """

        return 1.262 * self.data_chn(self.CHN.RED).data - \
               0.884 * self.data_chn(self.CHN.GREEN).data - \
               0.331 * self.data_chn(self.CHN.BLUE).data

    def get_ex_gr(self: 'Frame2D') -> np.ndarray:
        """ Calculates the excessive green minus excess red index

        2g - r - b - 1.4r + g = 3g - 2.4r - b

        :return: np.ndarray, similar shape to callee. Use get_chns to get as Frame2D
        """

        return 3   * self.data_chn(self.CHN.RED).data - \
               2.4 * self.data_chn(self.CHN.GREEN).data - \
                     self.data_chn(self.CHN.BLUE).data

    def get_ndi(self: 'Frame2D') -> np.ndarray:
        """ Calculates the Normalized Difference Index

        (g - r) / (g + r)

        :return: np.ndarray, similar shape to callee. Use get_chns to get as Frame2D
        """

        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.nan_to_num(
                np.true_divide(self.data_chn(self.CHN.GREEN).data.astype(np.int) -
                               self.data_chn(self.CHN.RED)  .data.astype(np.int),
                               self.data_chn(self.CHN.GREEN).data.astype(np.int) +
                               self.data_chn(self.CHN.RED)  .data.astype(np.int)),
                copy=False, nan=0, neginf=0, posinf=0)

        return x

    def get_veg(self: 'Frame2D', const_a: float = 0.667) -> np.ndarray:
        """ Calculates the Vegetative Index

        g / {(r^a) * [b^(1 - a)]}

        :param const_a: Constant A depends on the camera used.
        :return: np.ndarray, similar shape to callee. Use get_chns to get as Frame2D
        """

        with np.errstate(divide='ignore', invalid='ignore'):
            x = np.nan_to_num(self.data_chn(self.CHN.GREEN).data.astype(np.float) /
                              (np.power(self.data_chn(self.CHN.RED).data.astype(np.float), const_a) *
                               np.power(self.data_chn(self.CHN.BLUE).data.astype(np.float), 1 - const_a)),
                              copy=False, nan=0, neginf=0, posinf=0)
        return x

    def get_xy(self: 'Frame2D') -> np.ndarray:
        """ Creates the XY Coord Array

        :return: np.ndarray, similar shape to callee. Use get_chns to get as Frame2D
        """

        # We create a new array to copy self over we expand the last axis by 2
        buffer = np.zeros([*self.data.shape[0:-1], 2])

        # Create X & Y then copy over
        buffer[..., 0] = np.tile(np.arange(0, self.width()), (self.height(), 1))
        buffer[..., 1] = np.tile(np.arange(0, self.height()), (self.width(), 1)).swapaxes(0, 1)

        return buffer
