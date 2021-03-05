import unittest

from tests.base.D2Fixture.test_fixture_spec import TestD2FixtureSpec


class TestSpecChannel(TestD2FixtureSpec):

    def test_labels(self):
        """ Verify labels are correctly initialized"""
        f = self.frame
        c = f.CHN
        self.assertEqual(0, f.labels[c.RED])
        self.assertEqual(1, f.labels[c.GREEN])
        self.assertEqual(2, f.labels[c.BLUE])
        self.assertEqual(3, f.labels[c.RED_EDGE])
        self.assertEqual(4, f.labels[c.NIR])

    def test_get_present(self):
        """ Gets the channels that is already present. Synonymous with data_chn """
        f = self.frame
        c = f.CHN
        self.assertEqual(0, f.get_chns(self_=False, chns=[c.RED])     .labels[c.RED])
        self.assertEqual(0, f.get_chns(self_=False, chns=[c.GREEN])   .labels[c.GREEN])
        self.assertEqual(0, f.get_chns(self_=False, chns=[c.BLUE])    .labels[c.BLUE])
        self.assertEqual(0, f.get_chns(self_=False, chns=[c.RED_EDGE]).labels[c.RED_EDGE])
        self.assertEqual(0, f.get_chns(self_=False, chns=[c.NIR])     .labels[c.NIR])

    def test_get_calculate(self):
        """ Calculates channels that are absent """
        f = self.frame
        c = f.CHN
        g = f.get_chns(self_=False, chns=[c.NDVI])  # This should give us the HSV Channels only

        self.assertEqual(0, g.data_chn([c.NDVI]).labels[c.NDVI])

        # NIR isn't present anymore due to self_=False
        self.assertRaises(KeyError, g.data_chn, [c.NIR])

    def test_shape(self):
        """ Tests the data shape and orientation """
        f = self.frame
        self.assertEqual((f.height(), f.width(), len(f.labels)), f.shape)


if __name__ == '__main__':
    unittest.main()
