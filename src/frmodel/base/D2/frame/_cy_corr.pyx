import numpy as np
cimport numpy as np
cimport cython
ctypedef np.uint8_t DTYPE_t8
ctypedef np.uint16_t DTYPE_t16
ctypedef np.uint32_t DTYPE_t32
from cython.parallel cimport prange
from tqdm import tqdm

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_corr(np.ndarray[DTYPE_t8, ndim=3] a,
            np.ndarray[DTYPE_t8, ndim=3] b,
            unsigned int radius,
            bint verbose):
    """ Entropy Calculation in Cython

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

    cdef DTYPE_t8 [:, :, :] a_view = a
    cdef DTYPE_t8 [:, :, :] b_view = b

    cdef unsigned short w_size = radius * 2 + 1
    cdef unsigned short wi_rows = (<unsigned int> a.shape[0]) - w_size + 1
    cdef unsigned short wi_cols = (<unsigned int> a.shape[1]) - w_size + 1
    cdef short wi_r = 0, wi_c = 0

    cdef unsigned int w_rows = (<unsigned int> a.shape[0]) - w_size - 1
    cdef unsigned int w_cols = (<unsigned int> a.shape[1]) - w_size - 1
    cdef short w_r = 0, w_c = 0

    cdef unsigned char w_channels = <char>a.shape[2]
    cdef char w_ch = 0

    out_v = np.zeros(shape=(wi_rows, wi_cols, w_channels), dtype=np.float)
    cdef double [:, :, :] out_v_view = out_v

    cdef double mean_a = 0.0
    cdef double mean_b = 0.0
    cdef double mean_a2 = 0.0
    cdef double mean_b2 = 0.0

    cdef double del_a = 0.0
    cdef double del_b = 0.0
    cdef double del_ab = 0.0

    cdef double std_a = 0.0
    cdef double std_b = 0.0

    # For each valid window row and column:
    #   We loop through channels, then rows, then columns of each window
    #       In each window we throw the value into a GLCM 65535 long array
    #       In that window, we then loop through the array and square sum
    #       We slot this value into the corresponding cell in entropy_ar

    # The outer loop is required for verbose tqdm
    # prange will not work if there are any Python objects within
    # So the options are exclusive, but verbose is more important
    for wi_r in tqdm(range(wi_rows), disable=not verbose, desc="Correlation Progress"):
        for wi_c in range(wi_cols):
            for w_ch in range(w_channels):
                # For each channel we need to calculate the auxilary values

                mean_a  = 0
                mean_b  = 0
                mean_a2 = 0
                mean_b2 = 0
                del_ab  = 0

                for w_r in prange(w_size, nogil=True, schedule='dynamic'):
                    for w_c in prange(w_size, schedule='dynamic'):
                        mean_a += <int>a_view[wi_r + w_r, wi_c + w_c, w_ch]
                        mean_b += <int>b_view[wi_r + w_r, wi_c + w_c, w_ch]

                        mean_a2 += <int>a_view[wi_r + w_r, wi_c + w_c, w_ch] ** 2
                        mean_b2 += <int>b_view[wi_r + w_r, wi_c + w_c, w_ch] ** 2

                mean_a /= (w_size ** 2)
                mean_b /= (w_size ** 2)

                mean_a2 /= (w_size ** 2)
                mean_b2 /= (w_size ** 2)

                std_a = (mean_a2 - mean_a ** 2) ** 0.5
                std_b = (mean_b2 - mean_b ** 2) ** 0.5

                for w_r in prange(w_size, nogil=True, schedule='dynamic'):
                    for w_c in prange(w_size, schedule='dynamic'):
                        del_a = <int>a_view[wi_r + w_r, wi_c + w_c, w_ch] - mean_a
                        del_b = <int>b_view[wi_r + w_r, wi_c + w_c, w_ch] - mean_b
                        del_ab += del_a * del_b


                if std_a * std_b == 0:
                    del_ab = np.nan
                else:
                    del_ab /= std_a * std_b

                out_v_view[wi_r, wi_c, w_ch] = del_ab

    return out_v


