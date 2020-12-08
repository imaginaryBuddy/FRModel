import numpy as np
cimport numpy as np
cimport cython
ctypedef np.uint16_t DTYPE_t16
ctypedef np.uint32_t DTYPE_t32
from cython.parallel cimport prange
from tqdm import tqdm


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_glcm(np.ndarray[DTYPE_t16, ndim=3] c,
               unsigned int radius,
               bint verbose):
    """ GLCM Calculation in Cython. (Deprecated for FRModel)

    This is exceptionally fast.
    However, because we use a very large buffer, we require a very large amount of RAM.

    This is not useful in practise due to this limitation, however, could be optimized using sparse matrices.

    The large RAM usage comes from using a 65536 == (256 ** 2) pre-made array to hold GLCM pair values. All of these
    arrays must be made before the execution of the algorithm hence it cannot be directly optimized.

    Requirements:
    Array C is defined as 256 * A + B.
    Array C Must be of np.uint16.
    This is to get a 1-to-1 unique tuple-like mapping.

    Limitations:
    Array Values cannot exceed 255, excluding 255.
    Array Values must be 3 dim, last dim will be iterated through
    Entropy result will overflow on > unsigned int
    Maximum GLCM occurrence cannot exceed 65535

    Verbose, whether to output progress with a progress bar

    Uses entropy_256_2dim for looping last dimension

    """

    # O O O           X X O   Window Size = 2
    # O O O if 2x2 -> X X O : Window I Rows = 2
    # O O O           O O O   Window I Cols = 2
    #
    # Window I Rows defines how many possible windows can be slid on the frame vertically.
    # Same with Columns
    #
    # Within each Window, we define
    #
    # X X  : Window Rows = 2
    # X X    Window Cols = 2
    #
    # This is the explicit number of rows and columns the window has

    cdef DTYPE_t16 [:, :, :] c_view = c

    cdef unsigned short w_size = radius * 2 + 1
    cdef unsigned short wi_rows = (<unsigned int> c.shape[0]) - w_size + 1
    cdef unsigned short wi_cols = (<unsigned int> c.shape[1]) - w_size + 1
    cdef short wi_r = 0, wi_c = 0

    cdef unsigned int w_rows = (<unsigned int> c.shape[0]) - w_size - 1
    cdef unsigned int w_cols = (<unsigned int> c.shape[1]) - w_size - 1
    cdef short w_r = 0, w_c = 0

    cdef unsigned char w_channels = <char>c.shape[2]
    cdef char w_ch = 0

    glcm_v = np.zeros(shape=(65536, wi_rows, wi_cols, w_channels), dtype=np.uint16)
    cdef unsigned short [:, :, :, :] glcm_v_view = glcm_v

    # For each valid window row and column:
    #   We loop through channels, then rows, then columns of each window
    #       In each window we throw the value into a GLCM 65535 long array
    #       In that window, we then loop through the array and square sum
    #       We slot this value into the corresponding cell in entropy_ar

    # The outer loop is required for verbose tqdm
    # prange will not work if there are any Python objects within
    # So the options are exclusive, but verbose is more important
    for wi_r in tqdm(range(wi_rows), disable=not verbose, desc="Entropy Progress"):
        for wi_c in prange(wi_cols, nogil=True, schedule='dynamic'):
            for w_ch in prange(w_channels, schedule='dynamic'):
                for w_r in prange(w_size, schedule='dynamic'):
                    for w_c in prange(w_size, schedule='dynamic'):
                        glcm_v_view[<int>c_view[wi_r + w_r, wi_c + w_c, w_ch], wi_r, wi_c, w_ch] += 1

    return glcm_v


