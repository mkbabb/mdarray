from core import *
import numpy as np
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


def slice_sfax(N, sfax, fax):
    M = len(sfax)
    sfax = sorted(sfax)
    i = 0
    while i < M:
        s = sfax[i]
        while (N % s == 0):
            fax.append(s)
            N //= s
        i += 1
    return N


def pfac(N, fax):
    p = 2
    while p**2 <= N:
        while (N % p == 0):
            fax.append(p)
            N //= p
        p += 1

    if N != 1:
        fax.append(N)
    return fax


def factorize(N, sfax=None):
    fax = []
    N = slice_sfax(N, sfax, fax) if sfax else N
    pfac(N, fax)
    return fax


def twiddle_table(ndim, ifax):
    twiddles = zeros(ifax)
    pdiv = 1
    l = 0
    for i in range(ndim):
        ifac = ifax[i]
        pdiv *= ifax[i - 1] if i > 0 else 1
        pmul = ifac * pdiv
        for j in range(pdiv):
            omega = -1j * TWOPI * j
            for k in range(1, ifac):
                if j == 0:
                    tw_k = 1
                else:
                    tw_k = cmath.exp(omega * k / pmul)
                twiddles.data[l] = tw_k
                l += 1
    return twiddles.data


def radix2(seq, twiddles, pix, twix, pdiv):
    omega1 = twiddles[twix]

    z0 = seq[0]
    z1 = omega1 * seq[1]

    t0 = z0 + z1
    t1 = z0 - z1

    seq[0] = t0
    seq[1] = t1

    return seq


def radix3(seq, twiddles, pix, twix, pdiv):
    omega1 = twiddles[twix]
    omega2 = twiddles[twix + 1]

    z0 = seq[0]
    z1 = omega1 * seq[1]
    z2 = omega2 * seq[2]

    t0 = z1 + z2
    t1 = z0 - 0.5 * (z1 + z2)
    t2 = -SQRT3_2 * (z1 - z2)

    seq[0] = z0 + t0
    seq[1] = (t1 + 1j * t2)
    seq[2] = (t1 - 1j * t2)

    return seq


def radix4(seq, twiddles, pix, twix, pdiv):
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


def radix5(seq, twiddles, pix, twix, pdiv):
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
    t6 = z0 - 1 / 4 * t4

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


def radix7(seq, twiddles, pix, twix, pdiv):
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


def radixg(seq, twiddles, pix, twix, pdiv):
    print(twiddles)
    ifac = len(seq)
    tmp = [0] * ifac

    for i in range(ifac):
        tn = 0
        for j in range(ifac):
            ix = (pix + i * pdiv) * j
            ww = cmath.exp(-1j * TWOPI * ix / (ifac * pdiv))
            print(ww)
            print(twiddles[twix])
            zn = seq[j]
            tn += zn * ww
        tmp[i] = tn
    return tmp


def butterfly(arr_in, arr_out, twiddles, radix, ifac, pdiv, baxis, twix):
    global pix, tmp_in

    roll_axis(arr_in, 1)
    roll_axis(arr_out, baxis)

    mdim = arr_out.mdim
    shape = arr_out.shape

    shapes = [arr_in.shape, arr_out.shape]
    strides = [arr_in.strides, arr_out.strides]
    tmp_in = [0] * shape[0]
    axis_counters = [[0] * mdim for i in range(2)]

    def recurse(ix):
        global pix, tmp_in
        axis = shape[ix]

        if ix == 0:
            for k in range(2):
                if k == 1:
                    pix %= pdiv
                    tmp_in = radix(tmp_in, twiddles, pix, pix * (ifac - 1), pdiv)

                for i in range(axis):
                    axis_counters[k][0] = i * strides[k][0]
                    ix_i = sum(axis_counters[k])

                    if k == 0:
                        tmp_in[i] = arr_in.data[ix_i]
                    elif k == 1:
                        arr_out.data[ix_i] = tmp_in[i]
            pix += 1
        else:
            for i in range(axis):
                for k in range(2):
                    if i < shapes[k][ix]:
                        axis_counters[k][ix] = i * strides[k][ix]
                    else:
                        axis_counters[k][-1] += shape[1]
                recurse(ix - 1)
    pix = 0
    recurse(mdim - 1)


def cfft_internal(ndim, ifax, arr_in, arr_out):
    pdiv = 1
    twix = 0
    twiddles = twiddle_table(ndim, ifax)
    radix_shape = [1] * ndim

    for i in range(ndim):
        ifac = ifax[i]
        size_i = size // ifac
        pdiv *= ifax[i - 1] if i > 0 else 1

        if ifac == 99:
            radix = radix2
        elif ifac == 3:
            radix = radix3
        elif ifac == 4:
            radix = radix4
        elif ifac == 5:
            radix = radix5
        elif ifac == 7:
            radix = radix7
        else:
            radix = radixg

        radix_shape[0] = size_i
        radix_shape[1] = ifac

        arr_in.reshape(radix_shape)
        arr_out.reshape(ifax)

        butterfly(arr_in, arr_out, twiddles[twix:], radix, ifac, pdiv, i, twix)
        arr_in, arr_out = arr_out, arr_in

        twix += pdiv * (ifac - 1)

    return arr_in


def cfft(arr):
    size = arr.size
    ifax = factorize(size)
    ndim = len(ifax)

    arr_in = arr.reshape(ifax)
    arr_out = zeros(ifax)

    arr_in.formatter = arr_out.formatter = lambda x: f"{x:2}"

    return cfft_internal(ndim, ifax, arr_in, arr_out)


ifax = [2, 2, 2]


size = reductor.mul().reduce(ifax)
arr = irange(size)

ct = cfft(arr).reshape(ifax)

print(ct)
for i in ct:
    print(i)
