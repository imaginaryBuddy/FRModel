from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Union
from xml.etree import ElementTree

import gdal
import numpy as np
import pandas as pd
import utm
from laspy.file import File
from frmodel.base.D3.cloud._cloud_frame import _Cloud3DFrame


@dataclass
class Cloud3D(_Cloud3DFrame):

    f: File

    @staticmethod
    def from_las(las_path:str) -> Cloud3D:
        """ Loads a Cloud3D from a LAS file

        :param las_path: Path to a las file
        :return: A Cloud3D instance
        """
        return Cloud3D(File(las_path, mode='r'))

    def to_latlong(self):
        utm_ = np.vstack([self.f.X, self.f.Y, self.f.Z]).astype(np.float)
        fmin = self.f.header.min
        fmax = self.f.header.max
        for i in range(3):
            c = utm_[i]
            utm_[i] = (c - np.min(c)) / (np.max(c) - np.min(c)) * (fmax[i] - fmin[i]) + fmin[i]

        utm_[0], utm_[1] = utm.to_latlon(utm_[0], utm_[1], 48, 'N')

    def data(self, sample_size: Union[int, float] = None, transformed: bool = True):
        """ Gets the data as a NumPy Array

        Returns as a 'X', 'Y', 'Z', 'Red', 'Green', 'Blue' column array

        :param sample_size: The number of points to randomly pick.
            Integer if specifying number of points
            Float if specifying proportion of points
        :param transformed: If the points should be adjusted based on the las header data
        :return: NumPy Array
        """

        # Choice by Fraction
        if isinstance(sample_size, float):
            sample_size = int(len(self.f.points) * sample_size)

        data = np.random.choice(self.f.points, sample_size, False) if sample_size else self.f.points
        data2 = pd.DataFrame(data['point'][['X', 'Y', 'Z', 'red', 'green', 'blue']]).to_numpy()
        if transformed: data2 = Cloud3D._transform_data(data2, self.header)
        return deepcopy(data2)

    def df(self, sample_size: Union[int, float] = None, transformed: bool = True):
        """ Gets the data as a Pandas DataFrame

        Columns returned are 'X', 'Y', 'Z', 'Red', 'Green', 'Blue'

        :param sample_size: The number of points to randomly pick.
            Integer if specifying number of points
            Float if specifying proportion of points
        :param transformed: If the points should be adjusted based on the las header data
        :return: Pandas DataFrame
        """
        d = self.data(sample_size, transformed)
        return pd.DataFrame(d, columns=['X', 'Y', 'Z', 'Red', 'Green', 'Blue'])

    @property
    def header(self):
        return self.f.header

    def close(self):
        self.f.close()

    @staticmethod
    def get_geo_info(geotiff_path: str):
        ds = gdal.Open(geotiff_path)
        xoffset, px_w, rot1, yoffset, px_h, rot2 = ds.GetGeoTransform()

        return dict(xoffset=xoffset, px_w=px_w, rot1=rot1,
                    yoffset=yoffset, px_h=px_h, rot2=rot2)

    def write_las(self, file_name: str, alt_header:File = None):
        """ Writes the current points in a las format

        Header used will be the same as the one provided during loading otherwise given.
        """

        f = File(file_name, mode='w', header=alt_header if alt_header else self.header)
        data = self.data(transformed=False)

        f.X = data[:, 0]
        f.Y = data[:, 1]
        f.Z = data[:, 2]

        f.Red   = data[:, 3]
        f.Green = data[:, 4]
        f.Blue  = data[:, 5]

        f.close()
        
    @staticmethod
    def _transform_data(data, header: File):
        """ Transforms data suitable for usage, from LAS format """
        return np.hstack([Cloud3D._transform_xyz(data[:, :3], header),
                          Cloud3D._transform_rgb(data[:, 3:])])

    @staticmethod
    def _inv_transform_data(data, header: File):
        """ Transforms data suitable for writing """
        return np.hstack([Cloud3D._inv_transform_xyz(data[:, :3], header),
                          Cloud3D._inv_transform_rgb(data[:, 3:])])

    @staticmethod
    def _transform_xyz(xyz, header: File):
        """ Transforms XYZ according to the header information

        This transforms XYZ into a workable, intended format for usage.
        """
        # noinspection PyUnresolvedReferences
        return xyz + [o / s for o, s in zip(header.offset, header.scale)]

    @staticmethod
    def _transform_rgb(rgb):
        """ Transforms RGB according to the header information

        This transforms RGB into 0 - 255
        """
        return rgb // (2 ** 8)

    @staticmethod
    def _inv_transform_xyz(xyz, header: File):
        """ Inverse Transforms XYZ according to the header information

        This inverse transforms XYZ according to header, intended for writing
        """
        # noinspection PyUnresolvedReferences
        return xyz - [o / s for o, s in zip(header.offset, header.scale)]

    @staticmethod
    def _inv_transform_rgb(rgb):
        """ Inverse Transforms RGB according to the header information

        This transforms RGB into 0 - 65535, intended for writing
        """
        return rgb * (2 ** 8)

    @staticmethod
    def _read_xml(xml_path):
        """ Reads XML and returns

        :param xml_path: Path to XML metadata
        :returns: Latitude, Longitude, Origin X, Y, Z respectively
        """
        root = ElementTree.parse(xml_path).getroot()
        return [float(i) for i in root[0].text[4:].split(",")] + [float(i) for i in root[1].text.split(",")]
