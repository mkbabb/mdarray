"""Hand-optimized radix butterflies for small primes.

These implement the short DFT algorithms from Temperton (1983), Table I.
Each butterfly operates on a length-n sequence with twiddle factor
premultiplication. The operation counts match Temperton's table:

    n | real muls | real adds
    2 |     0     |     4
    3 |     4     |    12
    4 |     0     |    16
    5 |    12     |    24
    7 |    36     |    48

The generic radixg handles arbitrary primes via O(n^2) DFT.
"""

from __future__ import annotations

import cmath

TWOPI = 6.283185307179586476925286766559005768394338798750

SQRT3_2 = 0.866025403784438646763723170752936183471402626905190314027
SQRT5_4 = 0.559016994374947424102293417182819058860154589902881431067
SIN_2PI_5 = 0.951056516295153572116439333379382143405698634125750222447
SIN_2PI_10 = 0.587785252292473129168705954639072768597652437643145991072

COS_2PI_7 = 0.623489801858733530525004884004239810632274730896
COS_4PI_7 = -0.222520933956314404288902564496794759466355568764
COS_6PI_7 = -0.900968867902419126236102319507445051165919162131

SIN_2PI_7 = 0.781831482468029808708444526674057750232334518708
SIN_4PI_7 = 0.974927912181823607018131682993931217232785800619
SIN_6PI_7 = 0.433883739117558120475768332848358754609990727787


def radix2(
    seq: list[complex],
    twiddles: list[complex],
    pix: int,
    twix: int,
    pdiv: int,
) -> list[complex]:
    """Radix-2 butterfly: 0 real multiplications, 4 real additions.

    Computes a 2-point DFT with twiddle premultiplication:
    z0, z1 = seq[0], omega1*seq[1]; output = [z0+z1, z0-z1].
    """
    omega1 = twiddles[twix]

    z0 = seq[0]
    z1 = omega1 * seq[1]

    seq[0] = z0 + z1
    seq[1] = z0 - z1

    return seq


def radix3(
    seq: list[complex],
    twiddles: list[complex],
    pix: int,
    twix: int,
    pdiv: int,
) -> list[complex]:
    """Radix-3 butterfly: 4 real multiplications, 12 real additions.

    Uses cos(2pi/3) = -1/2, sin(2pi/3) = sqrt(3)/2 to avoid
    explicit complex exponentials.
    """
    omega1 = twiddles[twix]
    omega2 = twiddles[twix + 1]

    z0 = seq[0]
    z1 = omega1 * seq[1]
    z2 = omega2 * seq[2]

    t0 = z1 + z2
    t1 = z0 - 0.5 * (z1 + z2)
    t2 = -SQRT3_2 * (z1 - z2)

    seq[0] = z0 + t0
    seq[1] = t1 + 1j * t2
    seq[2] = t1 - 1j * t2

    return seq


def radix4(
    seq: list[complex],
    twiddles: list[complex],
    pix: int,
    twix: int,
    pdiv: int,
) -> list[complex]:
    """Radix-4 butterfly: 0 real multiplications, 16 real additions.

    Exploits the fact that powers of i (the 4th root of unity) are
    {1, -i, -1, i}, eliminating all real multiplications.
    """
    omega1 = twiddles[twix]
    omega2 = twiddles[twix + 1]
    omega3 = twiddles[twix + 2]

    z0 = seq[0]
    z1 = omega1 * seq[1]
    z2 = omega2 * seq[2]
    z3 = omega3 * seq[3]

    t0 = z0 + z2
    t1 = z1 + z3
    t2 = z0 - z2
    t3 = z1 - z3

    seq[0] = t0 + t1
    seq[1] = t2 - 1j * t3
    seq[2] = t0 - t1
    seq[3] = t2 + 1j * t3

    return seq


def radix5(
    seq: list[complex],
    twiddles: list[complex],
    pix: int,
    twix: int,
    pdiv: int,
) -> list[complex]:
    """Radix-5 butterfly: 12 real multiplications, 24 real additions.

    Uses precomputed sin/cos constants for the 5th roots of unity
    to minimize operation count.
    """
    omega1 = twiddles[twix]
    omega2 = twiddles[twix + 1]
    omega3 = twiddles[twix + 2]
    omega4 = twiddles[twix + 3]

    z0 = seq[0]
    z1 = omega1 * seq[1]
    z2 = omega2 * seq[2]
    z3 = omega3 * seq[3]
    z4 = omega4 * seq[4]

    t0 = z1 + z4
    t1 = z2 + z3
    t2 = z1 - z4
    t3 = z2 - z3
    t4 = t0 + t1

    t5 = SQRT5_4 * (t0 - t1)
    t6 = z0 - 0.25 * t4

    t7 = t6 + t5
    t8 = t6 - t5

    t9 = SIN_2PI_5 * t2 + SIN_2PI_10 * t3
    t10 = SIN_2PI_10 * t2 - SIN_2PI_5 * t3

    seq[0] = z0 + t4
    seq[1] = t7 - 1j * t9
    seq[2] = t8 - 1j * t10
    seq[3] = t8 + 1j * t10
    seq[4] = t7 + 1j * t9

    return seq


def radix7(
    seq: list[complex],
    twiddles: list[complex],
    pix: int,
    twix: int,
    pdiv: int,
) -> list[complex]:
    """Radix-7 butterfly: 36 real multiplications, 48 real additions.

    Uses precomputed sin/cos constants for the 7th roots of unity.
    The symmetric structure of the DFT matrix reduces the 7-point
    transform to real linear combinations plus imaginary cross-terms.
    """
    omega1 = twiddles[twix]
    omega2 = twiddles[twix + 1]
    omega3 = twiddles[twix + 2]
    omega4 = twiddles[twix + 3]
    omega5 = twiddles[twix + 4]
    omega6 = twiddles[twix + 5]

    z0 = seq[0]
    z1 = omega1 * seq[1]
    z2 = omega2 * seq[2]
    z3 = omega3 * seq[3]
    z4 = omega4 * seq[4]
    z5 = omega5 * seq[5]
    z6 = omega6 * seq[6]

    t0 = z1 + z6
    t1 = z2 + z5
    t2 = z3 + z4
    t3 = z1 - z6
    t4 = z2 - z5
    t5 = z3 - z4

    t6 = z0 + COS_2PI_7 * t0 + COS_4PI_7 * t1 + COS_6PI_7 * t2
    t7 = z0 + COS_4PI_7 * t0 + COS_6PI_7 * t1 + COS_2PI_7 * t2
    t8 = z0 + COS_6PI_7 * t0 + COS_2PI_7 * t1 + COS_4PI_7 * t2

    t9 = SIN_2PI_7 * t3 + SIN_4PI_7 * t4 + SIN_6PI_7 * t5
    t10 = SIN_4PI_7 * t3 - SIN_6PI_7 * t4 - SIN_2PI_7 * t5
    t11 = SIN_6PI_7 * t3 - SIN_2PI_7 * t4 + SIN_4PI_7 * t5

    seq[0] = z0 + t0 + t1 + t2
    seq[1] = t6 - 1j * t9
    seq[2] = t7 - 1j * t10
    seq[3] = t8 - 1j * t11
    seq[4] = t8 + 1j * t11
    seq[5] = t7 + 1j * t10
    seq[6] = t6 + 1j * t9

    return seq


def radixg(
    seq: list[complex],
    twiddles: list[complex],
    pix: int,
    twix: int,
    pdiv: int,
) -> list[complex]:
    """Generic O(n^2) DFT butterfly for arbitrary prime radices.

    Used as fallback when no optimized butterfly exists for the given radix.
    Computes the DFT directly from the definition.
    """
    ifac = len(seq)
    tmp = [0j] * ifac

    for i in range(ifac):
        tn: complex = 0
        for j in range(ifac):
            ix = (pix + i * pdiv) * j
            ww = cmath.exp(-1j * TWOPI * ix / (ifac * pdiv))
            zn = seq[j]
            tn += zn * ww
        tmp[i] = tn

    for i in range(ifac):
        seq[i] = tmp[i]
    return seq
