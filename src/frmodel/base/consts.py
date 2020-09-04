class CONSTS:
    """ This class holds all the constants.

    This is to facilitate magic constants/numbers around the program.
    """
    class CHANNEL:
        RED = 0
        GREEN = 1
        BLUE = 2

    class AXIS:
        X = 0
        Y = 1
        Z = 2

""" Deprecated D_TYPE on branch destruct

D_TYPE: np.dtype = \
        np.dtype([(CONSTS.CHANNEL.RED,   'u1'),   # 0 - 255
                  (CONSTS.CHANNEL.GREEN, 'u1'),   # 0 - 255
                  (CONSTS.CHANNEL.BLUE,  'u1')])  # 0 - 255

"""