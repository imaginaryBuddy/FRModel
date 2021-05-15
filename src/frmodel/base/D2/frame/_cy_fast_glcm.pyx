import numpy as np
cimport numpy as np
cimport cython
ctypedef np.uint8_t DTYPE_t8
ctypedef np.uint16_t DTYPE_t16
ctypedef np.uint32_t DTYPE_t32
ctypedef np.float32_t DTYPE_ft32
from cython.parallel cimport prange
from tqdm import tqdm
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_fast_glcm(np.ndarray[DTYPE_t8, ndim=5] windows_i,
                 np.ndarray[DTYPE_t8, ndim=5] windows_j,
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


    assert windows_i.shape[0] == windows_j.shape[0], f"Fast GLCM Dim 0 Mismatch {windows_i.shape[0]}, {windows_j.shape[0]}"
    assert windows_i.shape[1] == windows_j.shape[1], f"Fast GLCM Dim 1 Mismatch {windows_i.shape[1]}, {windows_j.shape[1]}"
    assert windows_i.shape[2] == windows_j.shape[2], f"Fast GLCM Dim 2 Mismatch {windows_i.shape[2]}, {windows_j.shape[2]}"
    assert windows_i.shape[3] == windows_j.shape[3], f"Fast GLCM Dim 3 Mismatch {windows_i.shape[3]}, {windows_j.shape[3]}"
    assert windows_i.shape[4] == windows_j.shape[4], f"Fast GLCM Dim 4 Mismatch {windows_i.shape[4]}, {windows_j.shape[4]}"

    cdef DTYPE_t8 [:, :, :, :, :] i_v = windows_i
    cdef DTYPE_t8 [:, :, :, :, :] j_v = windows_j

    cdef unsigned short window_rows = <unsigned int> windows_i.shape[0]
    cdef unsigned short window_cols = <unsigned int> windows_i.shape[1]
    cdef short wi_r = 0, wi_c = 0

    cdef unsigned int cell_rows = <unsigned int> windows_i.shape[2]
    cdef unsigned int cell_cols = <unsigned int> windows_i.shape[3]
    cdef short c_r = 0, c_c = 0

    cdef unsigned char channels = <char>windows_i.shape[4]
    cdef char ch = 0
    cdef char i = 0
    cdef char j = 0
    cdef float glcm_val = 0
    cdef float mean_i_val = 0
    cdef float mean_j_val = 0
    cdef float var_i_val = 0
    cdef float var_j_val = 0

    # Finds the maximum out of the 2 windows
    cdef char glcm_max_value = np.max([windows_i, windows_j]) + 1

    glcm        = np.zeros([glcm_max_value, glcm_max_value,
                            window_rows, window_cols, channels], dtype=np.uint8)
    contrast    = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    correlation = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    asm         = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    mean        = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    mean_i      = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    mean_j      = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    var         = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    var_i       = np.zeros([window_rows, window_cols, channels], dtype=np.float32)
    var_j       = np.zeros([window_rows, window_cols, channels], dtype=np.float32)

    # i, j, wi_r, wi_c, ch
    cdef unsigned char  [:, :, :, :, :] glcm_v        = glcm
    cdef float          [:, :, :]       contrast_v    = contrast
    cdef float          [:, :, :]       correlation_v = correlation
    cdef float          [:, :, :]       asm_v         = asm
    cdef float          [:, :, :]       mean_v        = mean
    cdef float          [:, :, :]       mean_i_v      = mean_i
    cdef float          [:, :, :]       mean_j_v      = mean_j
    cdef float          [:, :, :]       var_v         = var
    cdef float          [:, :, :]       var_i_v       = var_i
    cdef float          [:, :, :]       var_j_v       = var_j

    glcm_v       [:, :, :, :, :] = 0
    contrast_v   [:, :, :]       = 0
    correlation_v[:, :, :]       = 0
    asm_v        [:, :, :]       = 0

    # print(window_rows, window_cols, channels, cell_rows, cell_cols)

    # The outer loop is required for verbose tqdm
    # prange will not work if there are any Python objects within
    # So the options are exclusive, but verbose is more important
    for wi_r in tqdm(range(window_rows), disable=not verbose,
                     desc="GLCM Construction Pass"):
        for wi_c in prange(window_cols, nogil=True, schedule='dynamic'):
            for ch in prange(channels, schedule='dynamic'):
                for c_r in prange(cell_rows, schedule='dynamic'):
                    for c_c in prange(cell_cols, schedule='dynamic'):
                        i = <char>i_v[wi_r, wi_c, c_r, c_c, ch]
                        j = <char>j_v[wi_r, wi_c, c_r, c_c, ch]
                        glcm_v[i, j, wi_r, wi_c, ch] += 1

    for wi_r in tqdm(range(window_rows), disable=not verbose,
                     desc="GLCM Contrast, ASM, Mean Pass"):
        for wi_c in prange(window_cols, nogil=True, schedule='dynamic'):
            for ch in prange(channels, schedule='dynamic'):
                for i in prange(glcm_max_value, schedule='dynamic'):
                    for j in prange(glcm_max_value, schedule='dynamic'):
                        glcm_val = glcm_v[i, j, wi_r, wi_c, ch] / glcm_max_value
                        if glcm_val != 0:
                            contrast_v [wi_r, wi_c, ch] += glcm_val * ((i - j) ** 2)
                            asm_v      [wi_r, wi_c, ch] += glcm_val ** 2
                            mean_i_v   [wi_r, wi_c, ch] += glcm_val * i
                            mean_j_v   [wi_r, wi_c, ch] += glcm_val * j

    for wi_r in tqdm(range(window_rows), disable=not verbose,
                     desc="GLCM Variance Pass"):
        for wi_c in prange(window_cols, nogil=True, schedule='dynamic'):
            for ch in prange(channels, schedule='dynamic'):
                for i in prange(glcm_max_value, schedule='dynamic'):
                    for j in prange(glcm_max_value, schedule='dynamic'):
                        glcm_val = glcm_v[i, j, wi_r, wi_c, ch] / glcm_max_value
                        mean_i_val = mean_i_v[wi_r, wi_c, ch]
                        mean_j_val = mean_j_v[wi_r, wi_c, ch]
                        if glcm_val != 0:
                            var_i_v[wi_r, wi_c, ch] += glcm_val * (i - mean_i_val) ** 2
                            var_j_v[wi_r, wi_c, ch] += glcm_val * (j - mean_j_val) ** 2

    for wi_r in tqdm(range(window_rows), disable=not verbose,
                     desc="GLCM Correlation, Mean, Variance Merge Pass"):
        for wi_c in prange(window_cols, nogil=True, schedule='dynamic'):
            for ch in prange(channels,              schedule='dynamic'):
                for i in prange(glcm_max_value,     schedule='dynamic'):
                    for j in prange(glcm_max_value, schedule='dynamic'):
                        glcm_val = glcm_v[i, j, wi_r, wi_c, ch] / glcm_max_value
                        mean_i_val = mean_i_v[wi_r, wi_c, ch]
                        mean_j_val = mean_j_v[wi_r, wi_c, ch]
                        var_i_val  = var_i_v[wi_r, wi_c, ch]
                        var_j_val  = var_j_v[wi_r, wi_c, ch]
                        if glcm_val != 0:
                            if var_i_val != 0 and var_j_val != 0:
                                correlation_v[wi_r, wi_c, ch] += glcm_val * (
                                    (i - mean_i_val) * (j - mean_j_val) / sqrt(var_i_val * var_j_val)
                                )
                        mean_v[wi_r, wi_c, ch] = (mean_i_val + mean_j_val) / 2
                        var_v[wi_r, wi_c, ch]  = (var_i_val + var_j_val) / 2

    return contrast, correlation, asm, mean, var


