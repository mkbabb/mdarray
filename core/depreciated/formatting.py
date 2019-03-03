from core.formatting import trim_string

__all__ = ["print_array"]


def print_array(arr, sep=', ', formatter=None):
    mdim = arr.mdim
    strides = arr.strides
    axis_counter = [0] * mdim

    if not formatter:
        formatter = lambda x: '{0}'.format(x)

    def recurse(ix):
        axis = arr.shape[ix]
        remaining_axes = mdim - ix

        s = ''
        if remaining_axes == mdim:
            for i in range(axis):
                axis_counter[0] = i * strides[0]
                ix_i = sum(axis_counter)

                arr_val = arr.data[ix_i]

                val = formatter(arr_val)
                s += (val + sep) if i < axis - 1 else val
        else:
            new_line = '\n' * (ix)
            for i in range(axis):
                axis_counter[ix] = i * strides[ix]

                val = recurse(ix - 1)

                s += ' ' * (remaining_axes) + val if i > 0 else val
                s += sep.strip() if i < axis - 1 else ''
                s += new_line if i != axis - 1 else ''

        if ix == 0:
            s = trim_string(s, sep)
        s = '[{0}]'.format(s)
        return s
    return recurse(mdim - 1)
