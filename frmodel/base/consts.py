class CONSTS:
    """ This class holds all the constants.

    This is to facilitate magic constants/numbers around the program.
    """
    class CHANNEL:
        RED = "R"
        GREEN = "G"
        BLUE = "B"

    class AXIS:
        X = 1  # Not sure why is it like this, may need to "fix" this.
        Y = 0  # Images are stored as height x width, e.g. 1080 x 1920.
        Z = 2