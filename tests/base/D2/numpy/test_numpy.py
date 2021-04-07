import unittest

import numpy as np

from frmodel.base import CONSTS
from tests.base.D2Fixture.test_fixture import TestD2Fixture


class TestNumPy(TestD2Fixture):
    
    c = CONSTS.CHN

    def test_slice(self):
        """ Tests the NumPy Slicing for the XY Dims """
        f = self.frame[10:20, 15:25]
        self.assertEqual(300, f.size())

    def test_channel_index(self):
        """ Tests indexing a channel """
        f = self.frame[0, 0, self.c.RED]
        self.assertEqual(1, f.size())

    def test_channel_multi_index(self):
        """ Tests indexing multiple channels """
        f = self.frame[0, 0, [self.c.RED, self.c.BLUE]]
        self.assertEqual(2, f.size())
        self.assertEqual((1, 1, 2), f.shape)

    def test_duplicate_channel(self):
        """ Tests if we can duplicate a channel """
        try:
            self.frame[0, 0, [self.c.RED, self.c.RED]]
        except KeyError:
            return
        self.fail()

    def test_step(self):
        """ Tests slicing with stepping of 2 """
        f = self.frame[0:10:2, 10:20:2]
        self.assertEqual(75, f.size())

    def test_ellipsis(self):
        """ Tests ellipsis for xy with channel indexing """
        f = self.frame[..., self.c.RED]
        self.assertEqual(self.frame.shape[:-1], f.shape[:-1])
        self.assertEqual(1, f.shape[-1])

    def test_bad_type(self):
        """ Tests rejection of Channel Bad Type """
        try:
            self.frame[0, 0, 0]
        except KeyError:
            return
        self.fail("Failed to reject Channel Dimension Bad Type.")

    def test_bad_channel_slice(self):
        """ Tests rejection of Channel Bad Type """
        try:
            self.frame[0, 0, self.c.RED:self.c.GREEN]
        except KeyError:
            return
        self.fail("Failed to reject Channel Dimension Bad Slicing.")

    def test_bad_channel_type_slice(self):
        """ Tests rejection of Channel Type Slicing """
        try:
            self.frame[0, 0, 0:1]
        except KeyError:
            return
        self.fail("Failed to reject Channel Dimension Bad Type Slicing.")

    def test_bad_absent_slice(self):
        """ Tests rejection of Channel Slicing """
        try:
            self.frame[0, 0, self.c.NDVI]
        except KeyError:
            return
        self.fail("Failed to reject Channel Dimension Missing Channel.")


if __name__ == '__main__':
    unittest.main()
