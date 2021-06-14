from typing import List


class CONSTS:
    """ This class holds all the constants.

    This is to facilitate magic constants/numbers around the program.
    """

    class CHN:

        # RGB
        X           = "X"
        Y           = "Y"
        Z           = "Z"
        XY          = (X, Y)
        XYZ         = (X, Y, Z)
        RED         = "R"  # 650 +- 16
        GREEN       = "G"  # 560 +- 16
        BLUE        = "B"  # 450 +- 16
        RGB         = (RED, GREEN, BLUE)
        HUE         = "H"
        SATURATION  = "S"
        VALUE       = "V"
        HSV         = (HUE, SATURATION, VALUE)
        NDI         = "NDI"
        EX_G        = "EX_G"
        MEX_G       = "MEX_G"
        EX_GR       = "EX_GR"
        VEG         = "VEG"

        # Spectral
        RED_EDGE    = "RE"   # 730 +- 16
        NIR         = "NIR"  # 840 +- 26
        RGBRENIR    = (RED, GREEN, BLUE, RED_EDGE, NIR)
        NDVI        = "NDVI"
        BNDVI       = "BNDVI"
        GNDVI       = "GNDVI"
        GARI        = "GARI"
        GLI         = "GLI"
        GBNDVI      = "GBNDVI"
        GRNDVI      = "GRNDVI"
        NDRE        = "NDRE"
        LCI         = "LCI"
        MSAVI       = "MSAVI"
        OSAVI       = "OSAVI"
        """
        NDWI        = "NDWI"
        
        ARVI        = "ARVI"
        BWDRVI
        CCCI
        CIgreen
        CIrededge
        CVI
        CI
        CTVI
        GDVI
        EVI
        EVI2"""

        class GLCM:
            @staticmethod
            def _head(pref, suf):
                return tuple(f"{pref}_{s}" for s in suf) if isinstance(suf, List) else f"{pref}_{suf}"

            @staticmethod
            def CON(x): return CONSTS.CHN.GLCM._head("CON", x)
            @staticmethod
            def COR(x): return CONSTS.CHN.GLCM._head("COR", x)
            @staticmethod
            def ASM(x): return CONSTS.CHN.GLCM._head("ASM", x)
            @staticmethod
            def MEAN(x): return CONSTS.CHN.GLCM._head("MEAN", x)
            @staticmethod
            def VAR(x): return CONSTS.CHN.GLCM._head("STDEV", x)

        class KMEANS:
            LABEL = "KM_LABEL"

        class MNL:
            BINARY = "MNL_BINARY"
            DISTANCE = "MNL_DISTANCE"
            PEAKS = "MNL_PEAKS"
            WATER = "MNL_WATER"
            CANNY = "MNL_CANNY"

    class BOUNDS:
        MAX_RGB = 256
        MIN_RGB = 0
        MAX_RGB_SPEC = 2**12
        MIN_RGB_SPEC = 0
        MAX_RENIR_SPEC = 2**14
        MIN_RENIR_SPEC = 0

        MAXS_RGBRENIR_SPEC = [MAX_RGB_SPEC, MAX_RGB_SPEC, MAX_RGB_SPEC, MAX_RENIR_SPEC, MAX_RENIR_SPEC]
        MINS_RGBRENIR_SPEC = [MIN_RGB_SPEC, MIN_RGB_SPEC, MIN_RGB_SPEC, MIN_RENIR_SPEC, MIN_RENIR_SPEC]
        MAXS_RGB = [MAX_RGB, MAX_RGB, MAX_RGB]
        MINS_RGB = [MIN_RGB, MIN_RGB, MIN_RGB]

    class AXIS:
        X = 0
        Y = 1
        Z = 2
