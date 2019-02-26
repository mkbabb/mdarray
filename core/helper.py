from functools import reduce


__all__ = ["update_dict", "get_strides", "make_mdim_shape",
           "swap_item", "roll_array",
           "pair_wise", "remove_extraneous_dims", "flatten_list"]


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
    strides = [0] * N
    strides[0] = init

    for i in range(N - 1):
        init *= shape[i]
        strides[i + 1] = init

    return strides


def flatten_shape(shape, order=-1):
    mdim = len(shape)
    if order < 0:
        order += mdim
    elif order == 0:
        return shape
    new_mdim = mdim - order
    new_shape = [0] * (mdim - order)
    
    for i in range(new_mdim):
        new_shape[i] = shape[i]
    red = 1
    
    for i in range(new_mdim - 1, mdim):
        red *= shape[i]
    new_shape[-1] = red
    return new_shape


def make_mdim_shape(shape, ndim):
    mdim = len(shape)
    diff = mdim - ndim
    if mdim < ndim:
        shape += [1] * (ndim - mdim)
    elif mdim > ndim:
        shape = flatten_shape(shape, diff)
    return shape


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


def pair_wise(arr1, arr2, func):
    buff = [0] * len(arr1)
    for n, i in enumerate(arr1):
        t = func(i, arr2[n])
        buff[n] = t
    return buff


def remove_extraneous_dims(arr):
    def recurse(arr):
        if len(arr) == 1:
            try:
                arr = recurse(arr[0])
                return arr
            except IndexError:
                return arr
        else:
            return arr

    return recurse(arr)


def flatten_list(arr, order=1):
    global shape, mdim
    shape = [len(arr)]
    mdim = 0
    def recurse(arr):
        global shape, mdim
        ndim = len(arr)
        buff = []
        mdim = 0
        for i in range(ndim):
            arr_i = arr[i]
            if isinstance(arr_i, list):
                buff_r = recurse(arr_i)
                M = len(arr_i)
                if len(shape) <= mdim + 1:
                    shape.insert(-1, M)
                mdim += 1
                buff += [buff_r] if mdim <= order else buff_r
            elif mdim != 0:
                buff += [arr_i]
        if mdim == 0:
            return arr
        else:
            return buff
    flt = recurse(arr)
    return flt, mdim, shape
