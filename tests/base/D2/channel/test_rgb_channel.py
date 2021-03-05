import unittest

from frmodel.base.D2.draw2D import Draw2D
from tests.base.D2Fixture.test_fixture import TestD2Fixture

import numpy as np


class TestRGBChannel(TestD2Fixture):

    def test_labels(self):
        """ Verify labels are correctly initialized"""
        f = self.frame
        c = f.CHN
        self.assertEqual(0, f.labels[c.RED])
        self.assertEqual(1, f.labels[c.GREEN])
        self.assertEqual(2, f.labels[c.BLUE])

    def test_data_present(self):
        """ Slices the data that is already present """
        f = self.frame
        c = f.CHN
        self.assertEqual(3, f.data_chn([c.RGB])  .shape[-1])
        self.assertEqual(1, f.data_chn([c.RED])  .shape[-1])
        self.assertEqual(1, f.data_chn([c.GREEN]).shape[-1])
        self.assertEqual(1, f.data_chn([c.BLUE]) .shape[-1])

    def test_data_absent(self):
        """ Attempts to get data that isn't there. Will throw an exception """
        f = self.frame
        c = f.CHN
        self.assertRaises(KeyError, f.data_chn, [c.EX_G])

    def test_get_present(self):
        """ Gets the channels that is already present. Synonymous with data_chn """
        f = self.frame
        c = f.CHN
        self.assertEqual(c.RED,   f.get_chns(self_=False, chns=[c.RED])  .labels(c.RED))
        self.assertEqual(c.GREEN, f.get_chns(self_=False, chns=[c.GREEN]).labels(c.GREEN))
        self.assertEqual(c.BLUE,  f.get_chns(self_=False, chns=[c.BLUE]) .labels(c.BLUE))

    def test_get_calculate(self):
        """ Calculates channels that are absent """
        f = self.frame
        c = f.CHN
        g = f.get_chns(self_=False, chns=[c.HSV])  # This should give us the HSV Channels only

        self.assertEqual(c.HUE,        g.data_chn([c.HUE]).labels(c.HUE))
        self.assertEqual(c.SATURATION, g.data_chn([c.SATURATION]).labels(c.SATURATION))
        self.assertEqual(c.VALUE,      g.data_chn([c.VALUE]).labels(c.VALUE))

        # Red isn't present anymore due to self_=False
        self.assertRaises(KeyError,    g.data_chn, [c.RED])

    def test_shape(self):
        """ Tests the data shape and orientation """
        f = self.frame
        self.assertEqual((f.height(), f.width(), len(f.labels)), f.shape)

if __name__ == '__main__':
    unittest.main()
