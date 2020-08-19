import os

_RSC = os.path.dirname(os.path.realpath(__file__)) + "/../rsc/"

_RAW = _RSC + "raws/"
_EXTRACT = _RSC + "samples/"

_RAW_VIDS = _RAW + "videos/"
_RAW_IMGS = _RAW + "_frames/"

EXTRACT_IMGS = _EXTRACT + "_frames/"
EXTRACT_VIDS = _EXTRACT + "videos/"

# Quick way to get a sample img
SAMPLE_CHESTNUT_0S_IMG   = EXTRACT_IMGS + "chestnut_0/frame00000ms.jpg"
SAMPLE_CHESTNUT_50S_IMG  = EXTRACT_IMGS + "chestnut_0/frame50000ms.jpg"
SAMPLE_CHESTNUT_100S_IMG = EXTRACT_IMGS + "chestnut_0/frame100000ms.jpg"

VID_CHESTNUT_0 = _RAW_VIDS + "chestnut_0.mp4"


