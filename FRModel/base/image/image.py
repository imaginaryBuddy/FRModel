from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
import numpy as np
from PIL import Image as PImage  # To avoid conflict.
from typing import Tuple
from array import array


@dataclass
class Image:
    """ This class will hold information about the image being used.

    Use from_image to read an image.

    This encapsulates the Pillow Image class to provide the required interfaces only.
    """

    img: PImage.Image

    class PartitionMethod(Enum):
        PAD = 0,
        CROP = 1,
        REMOVE = 2

    def partition(self,
                  window: Tuple = (100, 100),
                  stride: Tuple = (50, 50),
                  edge_method: PartitionMethod = PartitionMethod.PAD
                  ) -> array[Image]:
        """ Partitions the image into fixed sizes, with definable stride

        For example, if we have window = (100, 100), stride = (50, 50),
        the algorithm will move a 100x100px window along the image, extracting images

        :param edge_method: Defines the method to use to account for windows that exceed edges
        :param window: A Tuple of (x-axis, y-axis) size in px
        :param stride: A Tuple of (x-axis, y-axis) stride size in px
        """
        width = self.width()
        height = self.height()
        if edge_method == Image.PartitionMethod.PAD:
            # Default method of the cropping
            return [self.crop(w, h, w + window[0], h + window[1])
                    for w in range(0, width - stride[0], stride[0])
                    for h in range(0, height - stride[1], stride[1])]
        elif edge_method == Image.PartitionMethod.CROP:
            # Crops it further without the padding.
            return [self.crop(w, h, min(w + window[0], width), min(h + window[1], height))
                    for w in range(0, width - stride[0], stride[0])
                    for h in range(0, height - stride[1], stride[1])]
        elif edge_method == Image.PartitionMethod.REMOVE:
            return [self.crop(w, h, min(w + window[0], width), min(h + window[1], height))
                    for w in range(0, width - stride[0], stride[0])
                    for h in range(0, height - stride[1], stride[1])]

    def crop(self, left, upper, right, lower) -> Image:
        """ This crops the image, note the coordinate system starts from the top left."""
        return self.img.crop((left, upper, right, lower))

    @staticmethod
    def from_image(file_path: str) -> Image:
        """ Creates an instance using the file path. """
        return Image(PImage.open(file_path))

    def to_numpy(self) -> np.ndarray:
        """ Converts the current image to a numpy ndarray"""
        # noinspection PyTypeChecker
        return np.asarray(self.img)

    def save(self, file_path: str, **kwargs) -> None:
        """ Saves the current image file"""
        self.img.save(file_path, **kwargs)

    def size(self) -> Tuple[int, int]:
        """ Returns the size of the image as a tuple of (Width, Height) """
        return self.img.size

    def height(self) -> int:
        return int(self.img.height)

    def width(self) -> int:
        return int(self.img.width)

    def channel_red(self) -> Image:
        """ Gets the red channel of the Image """
        return Image(self.img.getchannel("R"))
    def channel_green(self) -> Image:
        """ Gets the green channel of the Image """
        return Image(self.img.getchannel("G"))
    def channel_blue(self) -> Image:
        """ Gets the blue channel of the Image """
        return Image(self.img.getchannel("B"))
    def channels(self):
        """ Splits the image into RGB Channels as a 3-size Tuple"""
        return (Image(i) for i in self.img.split())

