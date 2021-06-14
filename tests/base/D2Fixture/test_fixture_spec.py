import os
import unittest

from frmodel.base.D2 import Frame2D

_DIR = os.path.dirname(os.path.realpath(__file__))
_RSC = _DIR + "/../../../rsc"


class TestD2FixtureSpec(unittest.TestCase):

    @classmethod
    def setUp(cls) -> None:
        cls.frame = Frame2D.load(f"{_DIR}/sampletiff.npz")

if __name__ == '__main__':
    unittest.main()
