class CONSTS:
    """ This class holds all the constants.

    This is to facilitate magic constants/numbers around the program.
    """
    class CHANNEL:
        RED         = "R"
        GREEN       = "G"
        BLUE        = "B"
        HUE         = "H"
        SATURATION  = "S"
        VALUE       = "V"
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
            COR_RED   = "COR_R"
            COR_GREEN = "COR_G"
            COR_BLUE  = "COR_B"
            ENT_RED   = "ENT_R"
            ENT_GREEN = "ENT_G"
            ENT_BLUE  = "ENT_B"

    class AXIS:
        X = 0
        Y = 1
        Z = 2
