import unittest

import numpy as np

import core
import mdarray as md
from core import inf, nan


class test_ranges_pre_defined(unittest.TestCase):
    def test_zeros(self):
        list1 = [0] * 20
        arr1 = core.zeros([20])

        eq = (list1 == arr1.data)
        self.assertTrue(eq)

    def test_ones(self):
        list1 = [1] * 20
        arr1 = core.ones([20])

        eq = (list1 == arr1.data)
        self.assertTrue(eq)

    def test_full(self):
        list1 = [99.9] * 20
        arr1 = core.full([20], fill_value=99.9)

        eq = (list1 == arr1.data)
        self.assertTrue(eq)


class test_toarray(unittest.TestCase):
    def test_tomdarray_2d(self):
        list1 = [[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8]]
        arr1 = core.tomdarray(list1)
        arr2 = core.irange([3, 3])

        eq = (arr1.data == arr2.data)
        self.assertTrue(eq)

    def test_tomdarray_3d(self):
        list1 = [[[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8]],

                 [[9, 10, 11],
                  [12, 13, 14],
                  [15, 16, 17]]]
        arr1 = core.tomdarray(list1)
        arr2 = core.irange([3, 3, 2])

        eq = all((arr1 == arr2).data)
        self.assertEqual(eq, True)

    def test_tondarray_10d(self):
        shape = [2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2]
        arr1 = core.irange(shape)
        np_arr1 = core.tondarray(arr1)
        np_arr2 = np.arange(1024).reshape(shape)

        self.assertTrue(np.all(np_arr1 == np_arr2))


class test_ranges(unittest.TestCase):
    def test_irange(self):
        list1 = list(range(100))
        arr1 = core.irange(100)

        eq = (list1 == arr1.data)
        self.assertTrue(eq)

    def test_ilinear_range(self):
        list1 = [i for i in range(100)]

        arr1 = core.linear_range(0, 100, 100)

        eq = (list1 == arr1.data)
        self.assertTrue(eq)

    def test_flinear_range(self):
        denom = 100
        list1 = [round(i / denom, 8) for i in range(denom)]

        arr1 = core.linear_range(start=0.0, stop=1, size=denom)
        arr1.data = [round(i, 8) for i in arr1.data]

        eq = (list1 == arr1.data)
        self.assertTrue(eq)

    def test_ilog_range(self):
        base = 2
        list1 = [base**i for i in range(16)]

        arr1 = core.log_range(start=0, stop=16, size=16, base=base)

        eq = (list1 == arr1.data)
        self.assertTrue(eq)


class test_tiling_grid(unittest.TestCase):
    def test_make_mdim(self):
        arr1 = core.zeros([29, 23, 19, 1, 1, 1])
        arr2 = core.zeros([29, 23, 19])

        mdim1 = arr1.mdim
        core.make_mdim(arr2, mdim1)

        eq1 = (mdim1 == arr2.mdim)
        eq2 = (arr1.shape == arr2.shape)
        eq = (eq1 == eq2)
        self.assertTrue(eq)

    def test_sort_raxes(self):
        raxes1 = [0, 5, 2, 3]
        repts1 = [7, 8, 9, 10]
        mdim = 7
        core.sort_raxes(raxes1, repts1, mdim)

        raxes2 = [0, 1, 2, 3, 1, 5, 1]
        repts2 = [7, 1, 9, 10, 1, 8, 1]

        self.assertEqual(raxes1, raxes2)
        self.assertEqual(repts1, repts2)

    def test_repeat(self):
        arr1 = core.irange([2, 2, 1])
        raxes = [2, 0, 1]
        repts = [2, 2, 2]
        arr1 = core.repeat(arr1, raxes, repts)
        arr2 = core.tomdarray([[[0, 1, 0, 1],
                                [0, 1, 0, 1],
                                [2, 3, 2, 3],
                                [2, 3, 2, 3]],

                               [[0, 1, 0, 1],
                                [0, 1, 0, 1],
                                [2, 3, 2, 3],
                                [2, 3, 2, 3]]])

        eq = (arr1.data == arr2.data)
        self.assertTrue(eq)

    def test_dense_meshgrid(self):
        x = range(0, 3)
        y = range(1, 5)
        z = range(2, 7)

        x_arr = core.tomdarray(x).reshape([3, 1, 1])
        y_arr = core.tomdarray(y).reshape([1, 4, 1])
        z_arr = core.tomdarray(z).reshape([1, 1, 5])

        arrs = (x_arr, y_arr, z_arr)
        dgrid = core.dense_meshgrid(x, y, z)

        for i in range(len(dgrid)):
            eq_i = (arrs[i].data == dgrid[i].data)
            self.assertTrue(eq_i)


class test_broadcasting(unittest.TestCase):
    def test_generate_broadcast_shape(self):
        arr1 = core.zeros([1, 5, 4, 3, 2, 2])
        arr2 = core.zeros([6, 1, 4, 3, 2, 2])
        arr3 = core.zeros([6, 5, 1, 3, 2, 2])
        arr4 = core.zeros([6, 5, 4, 1, 2, 2])
        arr5 = core.zeros([6, 5, 4, 3, 1, 2])
        arr6 = core.zeros([6, 5, 4, 3, 2, 1])

        arr7 = core.zeros([6, 5, 4, 3, 2, 2])

        shape1, repts1 = core.generate_broadcast_shape(arr1, arr2, arr3,
                                                       arr4, arr5, arr6)
        shape2 = arr7.shape

        eq = (shape1 == shape2)
        self.assertTrue(eq)

    def test_broadcast_bnry(self):
        arr1 = core.linear_range(0, 8, 8).reshape([2, 2, 2])
        arr2 = core.linear_range(2, 4, 2).reshape([2])
        arr3 = core.tomdarray([[[2, 4],
                                [4, 6]],

                               [[6, 8],
                                [8, 10]]])

        bnry_func = lambda x, y: x + y
        arr2 = core.broadcast_bnry(arr1, arr2, func=bnry_func)

        eq = (arr2.data == arr3.data)
        self.assertTrue(eq)

    def test_broadcast_arrays(self):
        arr1 = core.irange(10).reshape([5, 1, 2])
        arr2 = core.irange(5)
        arr3 = core.irange(20).reshape([5, 2, 2])

        arr1, arr2, arr3 = core.broadcast_arrays(arr1, arr2, arr3)
        arr5 = core.tomdarray([[[0, 1, 2, 3, 4],
                                [0, 1, 2, 3, 4]],

                               [[0, 1, 2, 3, 4],
                                [0, 1, 2, 3, 4]]])

        eq2 = (arr2.data == arr5.data)
        self.assertTrue(eq2)

    def test_broadcast_topshape(self):
        arr1 = core.irange(3)
        shape1 = [3, 1, 1, 3]

        arr1 = core.broadcast_toshape(arr1, shape1)
        arr2 = core.tomdarray([[[[0, 1, 2]]],
                               [[[0, 1, 2]]],
                               [[[0, 1, 2]]]])

        eq = (arr1.data == arr2.data)
        self.assertTrue(eq)


unittest.main()
