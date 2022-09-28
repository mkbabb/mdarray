class MArray:
    def __init__(self):
        self.data = []
        self.mdim = 0
        self.shape = []
        self.size = 0
        self.strides = []
        self.axis_counter = []
        self.was_advanced = []

        self.pos = 0
        self.index = 0


def get_strides(mdim: int, shape: List[int]) -> List[int]:
    init = 1
    strides = [0] * mdim
    strides[0] = init

    for i in range(mdim - 1):
        init *= shape[i]
        strides[i + 1] = init
    return strides


def marray_advance(arr: MArray):
    for i in range(1, arr.mdim):
        if arr.axis_counter[i - 1] >= arr.stride_shape[i - 1]:
            arr.axis_counter[i - 1] = 0
            arr.axis_counter[i] += arr.strides[i]
            arr.was_advanced[i] = True
        else:
            arr.was_advanced[i] = False

    index = 0
    for i in range(arr.mdim):
        index += arr.axis_counter[i]

    arr.index = index
    arr.pos += 1

    return arr.index


def create_marray(mdim: int, shape: List[int]):
    arr = MArray()
    size = 1
    for i in range(mdim):
        size *= shape[i]

    arr.data = [0] * size
    arr.mdim = mdim
    arr.shape = shape
    arr.size = size
    arr.strides = get_strides(mdim, shape)
    arr.axis_counter = [0] * mdim
    arr.was_advanced = [False] * mdim

    return arr
