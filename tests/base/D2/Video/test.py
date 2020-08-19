import unittest

from frmodel.base.D2.video2D import Video2D


class VideoTest(unittest.TestCase):

    def test(self):

        """ Test not available due to required sample

        # Grab the video
        vid = Video2D.from_video("sample.mp4")

        # Grab the 1s and 2s frame. If fail, substitute with None.
        frames = vid.to_frames(offsets_msec=[1000, 2000], failure_default=None)

        # Grab the 1s frame's Green Channel
        green = frames[0].channel(CONSTS.CHANNEL.GREEN)

        green.save("out.png")
        """


if __name__ == '__main__':
    unittest.main()
