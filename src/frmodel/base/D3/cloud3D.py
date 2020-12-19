from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from xml.etree import ElementTree

import gdal
import numpy as np
import pandas as pd
from laspy.file import File


@dataclass
class Cloud3D:

    f: File
    lat: float
    long: float
    origin_x: float
    origin_y: float
    origin_z: float

    @staticmethod
    def from_las_xml(las_path:str, xml_path:str) -> Cloud3D:
        f = File(las_path, mode='r')

        lat, long, origin_x, origin_y, origin_z = Cloud3D._read_xml(xml_path)
        return Cloud3D(f, lat, long, origin_x, origin_y, origin_z)

    def data(self, sample_size=None, transformed=True):
        data = np.random.choice(self.f.points, sample_size, False) if sample_size else self.f.points
        data2 = pd.DataFrame(data['point'][['X', 'Y', 'Z', 'red', 'green', 'blue']]).to_numpy()
        if transformed: data2 = Cloud3D._transform_data(data2, self.header)
        return deepcopy(data2)

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
