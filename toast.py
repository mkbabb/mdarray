import core.creation
from MultiArray import MultiArray
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import reduce


# new_shape, repts = generate_broadcast_shape(MultiArray(
#     shape=[5, 1, 7]), MultiArray(shape=[5, 1, 1]), MultiArray(shape=[1, 1, 7]))

# print(new_shape, repts)

# def make_nested_list(arr: MultiArray) -> list:
#     tmp = []
#     nest = []

#     def init_nest(nest, depth):
#         if (depth == 0):
#             return
#         else:
#             nest.append([])
#             init_nest(nest[0], depth - 1)

#     def nest_item(item, nest, depth, count=0):
#         if (count == depth):
#             nest.append(item)
#         else:
#             nest_item([item], nest[0], depth, count + 1)

#     init_nest(nest, arr.mdim - 1)

#     for i in range(arr.size):
#         tmp.append(arr.data[arr.index])
#         next(arr)

#         ix = 0
#         for j in range(1, arr.mdim):
#             if arr.was_advanced[j]:
#                 ix += 1

#         if (i < arr.size - 1):
#             if (ix > 0):
#                 nest_item(tmp, nest, ix - 1)
#                 tmp = []
#             print(nest)

#     arr.at(0)
#     return nest


arr1 = core.creation.irange(27)

print(arr1)
arr = make_nested_list(arr1)
print(arr)
