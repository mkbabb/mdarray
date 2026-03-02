"""Twiddle factor precomputation for mixed-radix FFT.

Twiddle factors are the diagonal matrices D^{n_i}_{m_i} in Temperton's
factorization (Eq. 12). They encode the complex exponentials needed at each
stage of the butterfly.

The flat table is laid out by stage, then by group element, then by twiddle
index within the butterfly:

    For stage s with factor f_s and cumulative product pdiv = f_0 * ... * f_{s-1}:
        - There are pdiv groups, each needing (f_s - 1) twiddle factors.
        - tw[j, k] = exp(-2πi * j * k / (f_s * pdiv))
          for j in [0, pdiv), k in [1, f_s).
        - Total entries for stage s: pdiv * (f_s - 1).

    The offset into the flat table for stage s is the sum of entries from
    all previous stages.
"""

from __future__ import annotations

import cmath

TWOPI = 6.283185307179586476925286766559005768394338798750


def twiddle_table(ifax: list[int]) -> list[complex]:
    """Precompute twiddle factors for all stages of the mixed-radix FFT.

    Parameters
    ----------
    ifax : list[int]
        Factor list from ``factorize(N)``—the prime decomposition of the
        transform length, e.g. ``[2, 3, 5]`` for N=30.

    Returns
    -------
    list[complex]
        Flat list of precomputed twiddle factors, indexed by stage.
        For stage *s* with factor *f_s* and cumulative product
        ``pdiv = f_0 * ... * f_{s-1}``, the table contains ``pdiv * (f_s - 1)``
        entries: ``tw[j, k] = exp(-2πi·j·k / (f_s·pdiv))`` for
        j ∈ [0, pdiv), k ∈ [1, f_s).
    """
    total = 0
    pdiv = 1
    for i in range(len(ifax)):
        ifac = ifax[i]
        total += pdiv * (ifac - 1)
        pdiv *= ifac

    twiddles: list[complex] = [0j] * total

    pdiv = 1
    idx = 0
    for i in range(len(ifax)):
        ifac = ifax[i]
        pmul = ifac * pdiv
        for j in range(pdiv):
            omega = -1j * TWOPI * j
            for k in range(1, ifac):
                if j == 0:
                    tw_k: complex = 1 + 0j
                else:
                    tw_k = cmath.exp(omega * k / pmul)
                twiddles[idx] = tw_k
                idx += 1
        pdiv *= ifac
    return twiddles
