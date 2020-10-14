# cython: profile=True
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
ctypedef np.npy_uint8 DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def entropy_256_3dim_w(np.ndarray[DTYPE_t, ndim=3] a,
                       np.ndarray[DTYPE_t, ndim=3] b,
                       unsigned int radius):
    """ Entropy Calculation in Cython

    Limitations:
    Array Values cannot exceed 255, excluding 255.
    Array Values must be 3 dim, last dim will be iterated through
    Entropy result will overflow on > unsigned int
    Maximum GLCM occurrence cannot exceed 65535

    Uses entropy_256_2dim for looping last dimension

    """

    """
    O O O           X X O   Window Size = 2
    O O O if 2x2 -> X X O : Window I Rows = 2
    O O O           O O O   Window I Cols = 2
    
    Window I Rows defines how many possible windows can be slid on the frame vertically.
    Same with Columns
    
    Within each Window, we define 
    
    X X  : Window Rows = 2
    X X    Window Cols = 2
    
    This is the explicit number of rows and columns the window has
    """

    assert a.shape[0] == b.shape[0]
    assert a.shape[1] == b.shape[1]
    assert a.shape[2] == b.shape[2]

    cdef DTYPE_t [:, :, :] a_view = a
    cdef DTYPE_t [:, :, :] b_view = b

    cdef unsigned int w_size = radius * 2 + 1
    cdef unsigned int wi_rows = (<unsigned int> a.shape[0]) - w_size + 1
    cdef unsigned int wi_cols = (<unsigned int> a.shape[1]) - w_size + 1
    cdef size_t wi_r = 0, wi_c = 0

    cdef unsigned int w_rows = (<unsigned int> a.shape[0]) - w_size - 1
    cdef unsigned int w_cols = (<unsigned int> a.shape[1]) - w_size - 1
    cdef size_t w_r = 0, w_c = 0

    cdef unsigned char w_channels = <char>a.shape[2]
    cdef size_t w_ch = 0

    entropy_ar = np.zeros([wi_rows, wi_cols, w_channels], dtype=np.uint32)
    cdef unsigned int[:, :, :] entropy_view = entropy_ar

    cdef unsigned short glcm_size = 256
    cdef unsigned short glcm[256][256]
    cdef unsigned short [:, :] glcm_view = glcm
    cdef size_t glcm_r = 0, glcm_c = 0

    cdef unsigned int entropy = 0
    cdef unsigned char entropy_power = 2

    for wi_r in range(wi_rows):
        for wi_c in range(wi_cols):
            # Slide through possible window top lefts

            # Reset GLCM View
            glcm_view[:, :] = 0

            for w_ch in range(w_channels):
                for w_r in range(w_size):
                    for w_c in range(w_size):
                        glcm_view[a_view[wi_r + w_r, wi_c + w_c, w_ch],
                                  b_view[wi_r + w_r, wi_c + w_c, w_ch]] += 1 # Make into const?
                entropy = 0
                for glcm_r in range(glcm_size):
                    for glcm_c in range(glcm_size):
                        entropy += glcm_view[glcm_r, glcm_c] ** entropy_power
                entropy_view[wi_r, wi_c, w_ch] = entropy

    return entropy_ar