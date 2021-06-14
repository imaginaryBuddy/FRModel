from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import gdal
import numpy as np
import utm
from scipy import ndimage
from scipy.interpolate import griddata, CloughTocher2DInterpolator

from frmodel.base import CONSTS
from frmodel.base.D2 import Frame2D

if TYPE_CHECKING:
    from frmodel.base.D3 import Cloud3D

class _Cloud3DFrame(ABC):
    
    @abstractmethod
    def data(self, sample_size=None, transformed=True) -> np.ndarray:
        ...

    @staticmethod
    def _geotiff_to_latlong_ranges(geotiff_path:str) -> tuple:
        ds = gdal.Open(geotiff_path)

        width = ds.RasterXSize
        height = ds.RasterYSize
        gt = ds.GetGeoTransform()

        d2latmin, d2longmin = gt[3] + width * gt[4] + height * gt[5], gt[0]
        d2latmax, d2longmax = gt[3], gt[0] + width * gt[1] + height * gt[2]
        return (d2latmin, d2latmax), (d2longmin, d2longmax)

    def to_frame(self: 'Cloud3D',
                 geotiff_path: str,
                 shape: tuple,
                 samples: int = 100000):
        """ Converts this Cloud3D into a 2D Frame

        This algorithm uses geotiff metadata to fit the Cloud data onto it.

        :param geotiff_path: A Geo-referencable geotiff path
        :param shape: The expected shape, this is usually specified by the Frame2D.from_image_spec
        :param samples: The number of cloud samples to randomly sample for interpolation.
        :return:
        """
        # Extract the UTM Data
        utm_data = np.vstack([self.f.X, self.f.Y, self.f.Z]).astype(np.float)
        utm_min = np.asarray([*self.f.header.min])[..., np.newaxis]
        utm_max = np.asarray([*self.f.header.max])[..., np.newaxis]

        # Get the expected lat long ranges from our GEOTiff
        lat_range, lng_range = self._geotiff_to_latlong_ranges(geotiff_path)

        # For some odd reason, the UTM data isn't scaled correctly to the provided min-max
        # in the header. The incorrect scaled data is prev.
        # Hence, we need to rescale it to its appropriate axis' minmax
        utm_min_prev = np.min(utm_data, axis=1)[..., np.newaxis]
        utm_max_prev = np.max(utm_data, axis=1)[..., np.newaxis]
        utm_data = (utm_data - utm_min_prev) / (utm_max_prev - utm_min_prev) *\
                   (utm_max - utm_min) + utm_min

        # 0: Latitude, 1: lngitude
        utm_data[0], utm_data[1] = utm.to_latlon(utm_data[0], utm_data[1], 48, 'N')

        # The data now is in the correct lat lng.

        # Now, we need to trim out OOB lat lngs
        utm_data = utm_data[:,
                   np.logical_and.reduce((
                       utm_data[0] >= lat_range[0],
                       utm_data[0] <= lat_range[1],
                       utm_data[1] >= lng_range[0],
                       utm_data[1] <= lng_range[1],
                   ))]

        # Finally, we use the provided lat-lng ranges to scale to the shape
        lat_offset = lat_range[0]
        lat_scale = shape[0] / (lat_range[1] - lat_range[0])
        lng_offset = lng_range[0]
        lng_scale = shape[1] / (lng_range[1] - lng_range[0])

        utm_data[0] = (utm_data[0] - lat_offset) * lat_scale
        utm_data[1] = (utm_data[1] - lng_offset) * lng_scale

        rand = np.random.choice(np.arange(0, utm_data.shape[1]), samples if samples else utm_data.shape[1], replace=False)
        x = utm_data[0][rand]
        y = utm_data[1][rand]
        z = utm_data[2][rand]
        X = np.arange(0, shape[0])
        Y = np.arange(0, shape[1])
        XM, YM = np.meshgrid(X, Y)  # 2D grid for interpolation

        interp_z = CloughTocher2DInterpolator(list(zip(x, y)), z, rescale=True)
        Z = interp_z(XM, YM)
        Z = np.where(Z < 0, 0, Z)
        Z = np.nan_to_num(Z)
        # Not sure why the Y is inverted, something to do with the lat long
        return Frame2D(Z[:,::-1].T, labels=[Frame2D.CHN.Z])
    #
    # def to_frame1(self,
    #              sample_size=None, transformed=True,
    #              width=None, height=None,
    #              method: CONSTS.INTERP3D = CONSTS.INTERP3D.NEAREST,
    #              clamp_cubic: bool = True) -> Frame2D:
    #     """ Converts the Cloud3D into a Frame2D
    #
    #     :param sample_size: The number of random points to use for interpolation
    #     :param transformed: Whether to shift axis based on header information
    #     :param width: Width of resulting image
    #     :param height: Height of resulting image
    #     :param method: Method of interpolation, use CONSTS.INTERP3D for various methods
    #     :param clamp_cubic: Whether to clamp the cubic RGB output or not
    #     :return: A Frame2D with Z, R, G, B columns
    #     """
    #     # Grab array data and scale it down to desired height and width
    #     ar = self.data(sample_size, transformed)
    #     height_range  = np.max(ar[..., 0]) - np.min(ar[..., 0])
    #     width_range = np.max(ar[..., 1]) - np.min(ar[..., 1])
    #
    #     if height and not width:
    #         width = int(width_range / height_range * height)
    #     elif width and not height:
    #         height = int(height_range / width_range * width)
    #     else:
    #         raise Exception("Frame Height or Width must be specified")
    #
    #     ar[..., 0] = (ar[..., 0] - np.min(ar[..., 0])) / height_range * height
    #     ar[..., 1] = (ar[..., 1] - np.min(ar[..., 1])) / width_range * width
    #
    #     # Create grid to estimate
    #     grid_x, grid_y = np.mgrid[0:height, 0:width]
    #     grid = grid_x, grid_y
    #     method: str
    #     ar_intp = np.zeros(shape=(height, width, 4), dtype=np.float)
    #
    #     ar_intp[..., 0] = griddata(ar[..., 0:2], ar[..., 2], grid, method)
    #
    #     ar_intp[..., 1] = griddata(ar[..., 0:2], ar[..., 3], grid, method)
    #     ar_intp[..., 2] = griddata(ar[..., 0:2], ar[..., 4], grid, method)
    #     ar_intp[..., 3] = griddata(ar[..., 0:2], ar[..., 5], grid, method)
    #
    #     if method == CONSTS.INTERP3D.CUBIC and clamp_cubic:
    #         ar_intp[..., 1:4] = self.interp_sig_clamp(ar_intp[..., 1:4])
    #
    #     return Frame2D(ar_intp.swapaxes(0, 1), labels=(CONSTS.CHN.Z, *CONSTS.CHN.RGB))
    #
    # @staticmethod
    # def interp_sig_clamp(x: np.ndarray, alpha: float = 255, beta: float = 50):
    #     """ Used to clamp the RGB values based on the following formula
    #
    #     a / { 1 + exp [ - ( x - a / 2 ) / b ] }
    #
    #     :param x: Input
    #     :param alpha: Amplitude, makes clamping from [0, a]
    #     :param beta: Bend factor.
    #     """
    #
    #     return alpha / (1 + np.exp( - (x - alpha / 2) / beta))
    #
    #
