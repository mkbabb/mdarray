"""Prime factorization for FFT length decomposition."""

from __future__ import annotations

__all__ = ["factorize", "pfac", "slice_sfax"]


def slice_sfax(N: int, sfax: list[int], fax: list[int]) -> int:
    """Factor out specific small factors first (for optimal radix ordering)."""
    sfax = sorted(sfax)
    for s in sfax:
        while N % s == 0:
            fax.append(s)
            N //= s
    return N


def pfac(N: int, fax: list[int]) -> list[int]:
    """Trial division factorization."""
    p = 2
    while p * p <= N:
        while N % p == 0:
            fax.append(p)
            N //= p
        p += 1
    if N != 1:
        fax.append(N)
    return fax


def factorize(N: int, sfax: list[int] | None = None) -> list[int]:
    """Factorize N into prime factors for mixed-radix FFT.

    If sfax is provided, those factors are extracted first (in sorted order),
    which allows controlling the radix ordering.
    """
    fax: list[int] = []
    N = slice_sfax(N, sfax, fax) if sfax else N
    pfac(N, fax)
    return fax
