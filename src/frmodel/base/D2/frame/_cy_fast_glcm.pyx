import numpy as np
cimport numpy as np
cimport cython
ctypedef np.uint8_t DTYPE_t8
ctypedef np.uint16_t DTYPE_t16
ctypedef np.uint32_t DTYPE_t32
ctypedef np.float32_t DTYPE_ft32
from cython.parallel cimport prange
from tqdm import tqdm


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_fast_glcm(np.ndarray[DTYPE_t8, ndim=5] windows_a,
                 np.ndarray[DTYPE_t8, ndim=5] windows_b,
                 bint verbose):
    """ Fast GLCM Calculation in Cython. Inspired by Wang Jifei's Fast GLCM code.

    This MUST be implemented within controlled environments in Fast GLCM, else
    certain assumptions are not held.

    Windows must be 5 dims.
    (1) Window Row (2) Window Col (3) Cell Row (4) Cell Col (5) Channel

    Window values should not exceed uint16 size.

    The result array should be given.
    This is to make it easier to do Cython
    (1) Combination Array (2) Window Row (3) Window Col (4) Channel

    Result values should not exceed uint8 size.
    This will rarely happen, almost negligible.

    Results will only exceed if the window size > 255 and all values happen to be the same.

    """


    assert windows_a.shape[0] == windows_b.shape[0], f"Fast GLCM Dim 0 Mismatch {windows_a.shape[0]}, {windows_b.shape[0]}"
    assert windows_a.shape[1] == windows_b.shape[1], f"Fast GLCM Dim 1 Mismatch {windows_a.shape[1]}, {windows_b.shape[1]}"
    assert windows_a.shape[2] == windows_b.shape[2], f"Fast GLCM Dim 2 Mismatch {windows_a.shape[2]}, {windows_b.shape[2]}"
    assert windows_a.shape[3] == windows_b.shape[3], f"Fast GLCM Dim 3 Mismatch {windows_a.shape[3]}, {windows_b.shape[3]}"
    assert windows_a.shape[4] == windows_b.shape[4], f"Fast GLCM Dim 4 Mismatch {windows_a.shape[4]}, {windows_b.shape[4]}"

    cdef DTYPE_t8 [:, :, :, :, :] a = windows_a
    cdef DTYPE_t8 [:, :, :, :, :] b = windows_b

    cdef unsigned short window_rows = <unsigned int> windows_a.shape[0]
    cdef unsigned short window_cols = <unsigned int> windows_a.shape[1]
    cdef short wi_r = 0, wi_c = 0

    cdef unsigned int cell_rows = <unsigned int> windows_a.shape[2]
    cdef unsigned int cell_cols = <unsigned int> windows_a.shape[3]
    cdef short c_r = 0, c_c = 0

    cdef unsigned char channels = <char>windows_a.shape[4]
    cdef char ch = 0
    cdef char i = 0
    cdef char j = 0
    cdef int glcm_val = 0

    # Finds the maximum out of the 2 windows
    cdef char glcm_max_value = np.max([windows_a, windows_b]) + 1

    glcm        = np.zeros([glcm_max_value, glcm_max_value,
                            window_rows, window_cols, channels], dtype=np.uint8)
    contrast    = np.zeros([window_rows, window_cols, channels], dtype=np.uint16)
    correlation = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    asm         = np.zeros([window_rows, window_cols, channels], dtype=np.uint16)

    # i, j, wi_r, wi_c, ch
    cdef unsigned char  [:, :, :, :, :] glcm_v        = glcm
    cdef unsigned short [:, :, :]       contrast_v    = contrast
    cdef float          [:, :, :]       correlation_v = correlation
    cdef unsigned short [:, :, :]       asm_v         = asm

    glcm_v       [:, :, :, :, :] = 0
    contrast_v   [:, :, :]       = 0
    correlation_v[:, :, :]       = 0
    asm_v        [:, :, :]       = 0

    i_mean = np.mean(windows_a, axis=(2, 3), dtype=np.float32)
    j_mean = np.mean(windows_b, axis=(2, 3), dtype=np.float32)
    i_std  = np.std (windows_a, axis=(2, 3), dtype=np.float32)
    j_std  = np.std (windows_b, axis=(2, 3), dtype=np.float32)

    cdef float[:, :, :] i_mean_v = i_mean
    cdef float[:, :, :] j_mean_v = j_mean
    cdef float[:, :, :] i_std_v  = i_std
    cdef float[:, :, :] j_std_v  = j_std

    cdef float i_m = 0.0
    cdef float j_m = 0.0
    cdef float i_s = 0.0
    cdef float j_s = 0.0

    # print(window_rows, window_cols, channels, cell_rows, cell_cols)

    # The outer loop is required for verbose tqdm
    # prange will not work if there are any Python objects within
    # So the options are exclusive, but verbose is more important

    for wi_r in tqdm(range(window_rows), disable=not verbose, desc="GLCM First Pass"):
        for wi_c in prange(window_cols, nogil=True, schedule='dynamic'):
            for ch in prange(channels, schedule='dynamic'):
                for c_r in prange(cell_rows, schedule='dynamic'):
                    for c_c in prange(cell_cols, schedule='dynamic'):
                        i = <char>a[wi_r, wi_c, c_r, c_c, ch]
                        j = <char>b[wi_r, wi_c, c_r, c_c, ch]
                        glcm_v[i, j, wi_r, wi_c, ch] += 1

    # for wi_r in tqdm(range(window_rows), disable=not verbose, desc="GLCM First Pass"):
    #     for wi_c in range(window_cols):
    #         for ch in range(channels):
    #             for c_r in range(cell_rows):
    #                 for c_c in range(cell_cols):
    #                     i = <char>a[wi_r, wi_c, c_r, c_c, ch]
    #                     j = <char>b[wi_r, wi_c, c_r, c_c, ch]
    #                     print(i, j, wi_r, wi_c, c_r)
    #                     glcm_v[i, j, wi_r, wi_c, ch] += 1
    #
    # for wi_r in tqdm(range(window_rows), disable=not verbose, desc="GLCM Second Pass"):
    #     for wi_c in range(window_cols):
    #         for ch in range(channels):
    #             for i in range(glcm_max_value):
    #                 for j in range(glcm_max_value):
    #                     glcm_val = glcm_v[i, j, wi_r, wi_c, ch]
    #                     if glcm_val != 0:
    #                         contrast_v [wi_r, wi_c, ch] += glcm_val * ((i - j) ** 2)
    #                         asm_v      [wi_r, wi_c, ch] += glcm_val ** 2
    #
    #                         # Correlation
    #                         i_m = i_mean_v[wi_r, wi_c, ch]
    #                         j_m = j_mean_v[wi_r, wi_c, ch]
    #                         i_s = i_std_v [wi_r, wi_c, ch]
    #                         j_s = j_std_v [wi_r, wi_c, ch]
    #                         if i_s != 0 and j_s != 0:
    #                             correlation_v[wi_r, wi_c, ch] += <float>glcm_val * (
    #                                 (i - i_m) * (j - j_m) / (i_s * j_s)
    #                             )
    for wi_r in tqdm(range(window_rows), disable=not verbose, desc="GLCM Second Pass"):
        for wi_c in prange(window_cols, nogil=True, schedule='dynamic'):
            for ch in prange(channels,              schedule='dynamic'):
                for i in prange(glcm_max_value,     schedule='dynamic'):
                    for j in prange(glcm_max_value, schedule='dynamic'):
                        glcm_val = glcm_v[i, j, wi_r, wi_c, ch]
                        if glcm_val != 0:
                            contrast_v [wi_r, wi_c, ch] += glcm_val * ((i - j) ** 2)
                            asm_v      [wi_r, wi_c, ch] += glcm_val ** 2

                            # Correlation
                            i_m = i_mean_v[wi_r, wi_c, ch]
                            j_m = j_mean_v[wi_r, wi_c, ch]
                            i_s = i_std_v [wi_r, wi_c, ch]
                            j_s = j_std_v [wi_r, wi_c, ch]
                            #
                            if i_s != 0 and j_s != 0:
                                correlation_v[wi_r, wi_c, ch] += glcm_val * (
                                    (i - i_m) * (j - j_m) / (i_s * j_s)
                                )

    return contrast, correlation, asm


