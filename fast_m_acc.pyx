"""
This is a fast array accumulator to slightly speed up
the main message passing bottleneck - looping through all matches.
"""

import numpy as np
cimport numpy as np


# Type used for indexing into numpy arrays
DTYPE = np.uint32
ctypedef np.uint32_t DTYPE_t

# Type used for floating
FTYPE = np.float64
ctypedef np.float64_t FTYPE_t

def fast_m_acc(np.ndarray M, np.ndarray p_s, np.ndarray mu_s,
               np.ndarray p_gs, np.ndarray mu_gs, int N):
    assert (M.dtype == DTYPE and
            p_s.dtype == FTYPE and
            mu_s.dtype == FTYPE and
            p_gs.dtype == FTYPE and
            mu_gs.dtype == FTYPE)
    cdef int j
    cdef DTYPE_t winner, loser
    cdef FTYPE_t p_gs_j_0, p_gs_j_1, mu_gs_j_0, mu_gs_j_1
    cdef FTYPE_t p_s_winner, p_s_loser, mu_s_winner, mu_s_loser
    for j in range(N):
        # Assign vars
        winner = M[j, 0]
        loser = M[j, 1]
        p_gs_j_0 = p_gs[j, 0]
        p_gs_j_1 = p_gs[j, 1]
        mu_gs_j_0 = mu_gs[j, 0]
        mu_gs_j_1 = mu_gs[j, 1]

        p_s_winner = p_s[winner]
        p_s_loser = p_s[loser]
        mu_s_winner = mu_s[winner]
        mu_s_loser = mu_s[loser]

        # Update values
        p_s[winner] = p_s_winner + p_gs_j_0
        p_s[loser] = p_s_loser + p_gs_j_1
        mu_s[winner] = mu_s_winner + (mu_gs_j_0 * p_gs_j_0)
        mu_s[loser] = mu_s_loser + (mu_gs_j_1 * p_gs_j_1)
