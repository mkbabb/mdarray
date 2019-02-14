from core import *
import numpy as np


SQRT3_2 = 0.866025403784438646763723170752936183471402626905190314027


def radix2(seq, mul):
    omega1 = np.exp(-2j * np.pi * mul)

    z0 = seq[0]
    z1 = omega1 * seq[1]

    t0 = z0 + z1
    t1 = z0 - z1

    seq[0] = t0
    seq[1] = t1

    return seq


def radix3(seq, mul):
    omega1 = np.exp(1 * -2j * np.pi * mul)
    omega2 = np.exp(2 * -2j * np.pi * mul)

    z0 = seq[0]
    z1 = omega1 * seq[1]
    z2 = omega2 * seq[2]

    t0 = z1 + z2
    t1 = z0 - 1 / 2 * (z1 + z2)
    t2 = -SQRT3_2 * (z1 - z2)

    seq[0] = z0 + t0
    seq[1] = (t1 + 1j * t2)
    seq[2] = (t1 - 1j * t2)

    return seq


def _permute(ai, ao, ifac, seq):
    global ii, jj

    ix = [i for i in range(ii, ii + ifac)]
    ao[seq, ] = ai[ix, ]

    ii += ifac

    return seq


def _radix(radix, ifac, pdiv, seq):
    global jj

    if jj >= pdiv:
        jj = 0
    seq = radix(seq, jj / (pdiv * ifac))
    print(seq)

    jj += 1
    return seq


ifax = [3, 2, 2]


ndim = len(ifax)
size = reductor.mul().reduce(ifax)
print(size)


ixs = arange(size).reshape(ifax)

ai = arange(size)
ao = arange(size)


pdiv = 1
swch = 1
for i in range(ndim):
    ii = 0
    jj = 0
    ifac = ifax[i]
    size_i = size // ifac
    pdiv *= ifax[i - 1] if i > 0 else 1

    if ifac == 2:
        radix = radix2
    elif ifac == 3:
        radix = radix3

    ai.reshape([size_i, ifac])
    ai = reduce_array(ai, 1, lambda x: _radix(radix, ifac, pdiv, x))
    reduce_array(ixs, i, lambda x: _permute(ai, ao, ifac, x))
    ai, ao = ao, ai

ao.formatter = lambda x: f"{x:.2f}"
# print(ixs)
for i in ao:
    print(i)
