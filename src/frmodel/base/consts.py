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
            CON_RED   = "CON_R"
            CON_GREEN = "CON_G"
            CON_BLUE  = "CON_B"
            CON_RGB   = (CON_RED, CON_GREEN, CON_BLUE)
            COR_RED   = "COR_R"
            COR_GREEN = "COR_G"
            COR_BLUE  = "COR_B"
            COR_RGB   = (COR_RED, COR_GREEN, COR_BLUE)
            ENT_RED   = "ENT_R"
            ENT_GREEN = "ENT_G"
            ENT_BLUE  = "ENT_B"
            ENT_RGB   = (ENT_RED, ENT_GREEN, ENT_BLUE)

        class KMEANS:
            LABEL = "KM_LABEL"

    class AXIS:
        X = 0
        Y = 1
        Z = 2
