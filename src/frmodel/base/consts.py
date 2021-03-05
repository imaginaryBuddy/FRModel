from typing import Tuple, Iterable


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
                return tuple(f"{pref}_{s}" for s in suf) if isinstance(suf, Iterable) else f"{pref}_{suf}"

            @staticmethod
            def CON(x): return CONSTS.CHN.GLCM._head("CON", x)
            @staticmethod
            def COR(x): return CONSTS.CHN.GLCM._head("COR", x)
            @staticmethod
            def ENT(x): return CONSTS.CHN.GLCM._head("ENT", x)

        class KMEANS:
            LABEL = "KM_LABEL"

        class MNL:
            BINARY = "MNL_BINARY"
            DISTANCE = "MNL_DISTANCE"
            PEAKS = "MNL_PEAKS"
            WATER = "MNL_WATER"
            CANNY = "MNL_CANNY"

    class AXIS:
        X = 0
        Y = 1
        Z = 2

    class INTERP3D:
        NEAREST = 'nearest'
        LINEAR  = 'linear'
        CUBIC   = 'cubic'
