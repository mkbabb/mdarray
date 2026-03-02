"""Twiddle factor precomputation for mixed-radix FFT.

Twiddle factors are the diagonal matrices D^{n_i}_{m_i} in Temperton's
factorization (Eq. 12). They encode the complex exponentials needed at each
stage of the butterfly.
"""

from __future__ import annotations

import cmath

TWOPI = 6.283185307179586476925286766559005768394338798750


def twiddle_table(ndim: int, ifax: list[int]) -> list[complex]:
    """Precompute twiddle factors for all stages.

    For stage i with factor n_i and cumulative product pdiv = n_0 * ... * n_{i-1},
    we need (pdiv) groups of (n_i - 1) twiddle factors each:
        tw[j, k] = exp(-2*pi*i * j * k / (n_i * pdiv))
    for j in [0, pdiv), k in [1, n_i).

    Returns a flat list of complex twiddle factors.
    """
    total = 0
    pdiv = 1
    for i in range(ndim):
        ifac = ifax[i]
        total += pdiv * (ifac - 1)
        pdiv *= ifac

    twiddles: list[complex] = [0j] * total

    pdiv = 1
    idx = 0
    for i in range(ndim):
        ifac = ifax[i]
        pdiv *= ifax[i - 1] if i > 0 else 1
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
    return twiddles
