from typing import Tuple, Iterable


class CONSTS:
    """ This class holds all the constants.

    This is to facilitate magic constants/numbers around the program.
    """

    class CHN:
        X           = "X"
        Y           = "Y"
        XY          = (X, Y)
        RED         = "R"
        GREEN       = "G"
        BLUE        = "B"
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
        NDVI        = "NDVI"
        GNDVI       = "GNDVI"
        OSAVI       = "OSAVI"
        NDRE        = "NDRE"
        LCI         = "LCI"

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

    class AXIS:
        X = 0
        Y = 1
        Z = 2
