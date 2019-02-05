from functools import reduce

__all__ = ["update_dict", "get_strides", "swap_item", "roll_array",
           "pair_wise", "pair_wise_accumulate"]


def update_dict(d1, d2, recursive=True):
    if isinstance(d1, dict) and isinstance(d2, dict):
        for i, j in d1.items():
            if isinstance(j, dict):
                if recursive:
                    d2 = update_dict(j, d2)
                else:
                    d2.update({i: j})
            else:
                try:
                    d2.update({i: j})
                except ZeroDivisionError:
                    pass
    return d2


def get_strides(shape):
    N = len(shape)
    init = 1
    strides = [0]*N
    strides[0] = init

    for i in range(N - 1):
        init *= shape[i]
        strides[i+1] = init

    return strides


def swap_item(arr, ix1, ix2):
    t = arr[ix1]
    arr[ix1] = arr[ix2]
    arr[ix2] = t


def roll_array(arr, axis, iterations=1):
    ndim = len(arr)
    if axis == 0:
        return
    elif axis < 0:
        axis += ndim

    def recurse(ix):
        swap_item(arr, axis, ix)
        ix += 1
        if ix == axis:
            return
        else:
            recurse(ix)

    for i in range(iterations):
        recurse(0)


def pair_wise(a1, a2, func):
    tmp = [0]*len(a1)
    for n, i in enumerate(a1):
        t = func(i, a2[n])
        tmp[n] = t
    return tmp


def pair_wise_accumulate(a1, a2):
    return reduce(lambda x, y: x + y, pair_wise(a1, a2, lambda x, y: x*y))
