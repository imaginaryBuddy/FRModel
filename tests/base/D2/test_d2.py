import unittest

from rsc.samples.frames import chestnut_0


class TestD2(unittest.TestCase):

    @classmethod
    def setUp(cls) -> None:
        cls.frame = chestnut_0(0)
        cls.frame_window = cls.frame.split_xy(100)[0][0]
        cls.window = 100
        cls.channels = 3

if __name__ == '__main__':
    unittest.main()
