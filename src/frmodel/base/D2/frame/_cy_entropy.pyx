import numpy as np
cimport numpy as np
cimport cython
ctypedef np.npy_uint16 DTYPE_t
ctypedef np.npy_uint32 DTYPE_t32
from tqdm import tqdm

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_entropy(np.ndarray[DTYPE_t, ndim=3] c,
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

    cdef DTYPE_t [:, :, :] c_view = c

    cdef unsigned short w_size = radius * 2 + 1
    cdef unsigned short wi_rows = (<unsigned int> c.shape[0]) - w_size + 1
    cdef unsigned short wi_cols = (<unsigned int> c.shape[1]) - w_size + 1
    cdef size_t wi_r = 0, wi_c = 0

    cdef unsigned int w_rows = (<unsigned int> c.shape[0]) - w_size - 1
    cdef unsigned int w_cols = (<unsigned int> c.shape[1]) - w_size - 1
    cdef size_t w_r = 0, w_c = 0

    cdef unsigned char w_channels = <char>c.shape[2]
    cdef size_t w_ch = 0

    cdef np.ndarray[DTYPE_t32, ndim=3] entropy_ar = np.zeros([wi_rows, wi_cols, w_channels], dtype=np.uint32)
    cdef unsigned int[:, :, :] entropy_view = entropy_ar

    cdef unsigned short glcm_size = <unsigned short>65536
    cdef unsigned short glcm[65536]
    cdef unsigned short [:] glcm_view = glcm
    cdef size_t glcm_i = 0

    cdef unsigned int entropy = 0
    cdef unsigned char entropy_power = 2

    # For each valid window row and column:
    #   We loop through channels, then rows, then columns of each window
    #       In each window we throw the value into a GLCM 65535 long array
    #       In that window, we then loop through the array and square sum
    #       We slot this value into the corresponding cell in entropy_ar
    for wi_r in tqdm(range(wi_rows), disable=not verbose, desc="Entropy Progress"):
        for wi_c in range(wi_cols):
            # Slide through possible window top lefts
            glcm_view[:] = 0

            for w_ch in range(w_channels):
                for w_r in range(w_size):
                    for w_c in range(w_size):
                        glcm_view[c_view[wi_r + w_r, wi_c + w_c, w_ch]] += 1
                entropy = 0

                for glcm_i in range(glcm_size):
                    entropy += glcm_view[glcm_i] ** entropy_power
                entropy_view[wi_r, wi_c, w_ch] = entropy

    return entropy_ar
