import unittest

import numpy as np

from FRModel.base.consts import CONSTS
from rsc.samples.frames import chestnut_0


class ChannelTest(unittest.TestCase):

    def test(self):

        # Grab Sample
        frame = chestnut_0(0)

        # Grab the green channel
        green = frame.channel(CONSTS.CHANNEL.GREEN)

        # We can use expected numpy functions.
        # data grabs the underlying representation of Channel2D, np.ndarray
        np.sum(green.data)

        # We can slice the data like so, note that this will create a 100px wide slice.
        # green = green[:, 0:100] doesn't work. It'll be really hard to implement
        green.data = green.data[:, 0:100]

        # We can always save it, it's commented out so to not clutter tests
        # green.save('out.jpg')

if __name__ == '__main__':
    unittest.main()
