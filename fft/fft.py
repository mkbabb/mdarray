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
    print(ix, seq)
    ai[seq, ] = ao[ix, ].data

    ii += ifac

    return seq


def _radix(radix, ifac, pdiv, seq):
    global jj

    if jj >= pdiv:
        jj = 0
    seq = radix(seq, jj / (pdiv * ifac))

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
m2 = True if size % 2 == 0 else False
for i in range(ndim):
    ii = 0
    jj = 0
    p = 0 if ((i == ndim - 1) and not m2) else i
    ifac = ifax[i]
    size_i = size // ifac
    pdiv *= ifax[i - 1] if i > 0 else 1

    if ifac == 2:
        radix = radix2
    elif ifac == 3:
        radix = radix3

    ai.reshape([size_i, ifac])
    reduce_array(ai, 1, lambda x: _radix(radix, ifac, pdiv, x), ao)

    if i != 0 or ((i != ndim) and m2):
        reduce_array(ixs, p, lambda x: _permute(ai, ao, ifac, x), jiter=True)
    else:
        ai, ao = ao, ai


ao.formatter = lambda x: f"{x:.2f}"
# print(ixs)
for i in ao:
    print(i)
