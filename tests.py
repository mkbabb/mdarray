from functools import reduce

from mdarray import arange, full, mdarray, tomdarray, tondarray, zeros
from mdarray_indexing import expand_dims, expand_slice_array
from mdarray_types import nan, inf

shape = [5, 3, 2]
size = reduce(lambda x, y: x*y, shape)

arr = arange(size).reshape(shape)
np_arr = tondarray(arr)


myslice = [nan, ..., nan]

t0 = expand_dims(myslice, arr)
t1 = expand_slice_array(t0, arr)
print(t1)







