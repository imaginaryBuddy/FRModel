import unittest

from frmodel.base.D2 import Frame2D
import os

_RSC = os.path.dirname(os.path.realpath(__file__)) + "/../../../rsc"

class TestD2(unittest.TestCase):

    @classmethod
    def setUp(cls) -> None:
        cls.frame = Frame2D.from_image(f"{_RSC}/imgs/rgb/chestnut/frame10000ms.jpg")
        cls.frame_window = cls.frame.split_xy(100)[0][0]
        cls.window = 100
        cls._RSC = _RSC
        cls.channels = 3

if __name__ == '__main__':
    unittest.main()
