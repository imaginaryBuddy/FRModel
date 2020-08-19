""" Provides a simple way to quickly load a Frame to test with """

import os

from frmodel.frmodel.base import Frame2D

_RSC = os.path.dirname(os.path.realpath(__file__))

def chestnut_0(index: int = 0):
    """ Gets Frame (index * 10s) of the chestnut_0 video

    :param index: an integer between 0 - 11, inclusive.
    """
    assert 0 <= index <= 11, f"Index {index} not supported. Choose [0 - 11]"
    return Frame2D.from_image(f"{_RSC}/_frames/chestnut_0/frame{index}0000ms.jpg")
