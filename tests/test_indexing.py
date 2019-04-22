import unittest

import numpy as np

import core
import multiArray as ma
from core import inf, nan


class test_ravel(unittest.TestCase):
    def test_ravel_unravel(self):
        ixs1 = [0, 2, 4, 6, 1, 3, 5, 7, -2, -3, -4]
        ixs2 = [0, 2, 4, 6, 1, 3, 5, 7, 6, 5, 4]

        shape = [4, 2]

        raveled_ixs = core.ravel(ixs1, shape)
        unraveled_ixs = core.unravel(raveled_ixs, shape)

        eq = (unraveled_ixs == ixs2)
        self.assertTrue(eq)

    def test_expand_indicies_1d(self):
        arr1 = core.zeros([10, 8, 6, 4, 2, 1])
        mdim1 = arr1.mdim
        ixs1 = [inf] * mdim1
        ixs2 = [...] * mdim1
        ixs3 = [..., [0, 1, 2], [3, 4, 5], [0, 1], [0], inf]

        slc1, new_shape1, oned1 = core.expand_indicies(ixs1, arr1)
        slc2, new_shape2, oned2 = core.expand_indicies(ixs2, arr1)
        slc3, new_shape3, oned3 = core.expand_indicies(ixs3, arr1)

        slc3_test = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                     [0, 1, 2],
                     [3, 4, 5],
                     [0, 1],
                     [0],
                     [0]]
        new_shape3_test = [10, 3, 3, 2, 1, 1]

        eq1 = (slc1 == slc2)
        eq2 = (slc3 == slc3_test)
        eq3 = (oned1 == oned2 == oned3)
        eq4 = (new_shape1 == new_shape2 == arr1.shape)
        eq5 = (new_shape3 == new_shape3_test)

        self.assertTrue(eq1)
        self.assertTrue(eq2)
        self.assertTrue(eq3)
        self.assertTrue(eq4)
        self.assertTrue(eq5)

    def test_expand_indicies_md(self):
        arr1 = core.zeros([5, 5, 2])

        ixs1 = core.tomdarray([[0, 4]])
        ixs2 = core.tomdarray([[0],
                               [4]])
        ixs = [ixs1, ixs2]

        slc1, new_shape1, oned1 = core.expand_indicies(ixs, arr1)
        self.assertFalse(oned1)

    def test_slice_array(self):
        arr1 = core.irange([5, 5, 2])

        # Grab the corner pieces of arr1:
        ixs1 = [[0, 4], [0, 4], inf]

        arr_out1 = core.slice_array(ixs1, arr1, None, False)
        arr_out1_test = [0, 4, 20, 24, 25, 29, 45, 49]
        eq1 = (arr_out1.data == arr_out1_test)

        # Grab the first column from the first 3d axis:
        ixs2 = [0, inf, 0]
        arr_out2 = core.slice_array(ixs2, arr1, None, False)
        arr_out2_test = [0, 5, 10, 15, 20]
        eq2 = (arr_out2.data == arr_out2_test)

        # Grab the first, second and fourth rows from the entire array:
        ixs3 = [[0, 1, 3], inf, inf]

        arr_out3 = core.slice_array(ixs3, arr1, None, False)
        arr_out3_test = [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 16, 18, 20, 21,
                         23, 25, 26, 28, 30, 31, 33, 35, 36, 38, 40, 41, 43, 45, 46, 48]
        eq3 = (arr_out3.data == arr_out3_test)

        # Grab elements [0, 1, 5, 9, 10, 14, 15, 19,
        #               0, 1, 5, 9, 10, 14, 15, 19]
        # Showing off grabbing multiple of the same item.
        ixs4 = ([0, 1, 5, 9, 10, 14, 15, 19,
                 0, 1, 5, 9, 10, 14, 15, 19],)

        arr_out4 = core.slice_array(ixs4, arr1, None, False)
        arr_out4_test = [0, 1, 5, 9, 10, 14, 15, 19,
                         0, 1, 5, 9, 10, 14, 15, 19]
        eq4 = (arr_out4.data == arr_out4_test)

        # Grab two of the first row, two of the third row, all from the fourth column
        # of the entire array:
        ixs5 = [[0, 0, 2, 2], 3, inf]

        arr_out5 = core.slice_array(ixs5, arr1, None, False)
        arr_out5_test = [15, 15, 17, 17, 40, 40, 42, 42]
        eq5 = (arr_out5.data == arr_out5_test)

        # Grab the center of the entire array:
        ixs6 = [[1, 2, 3], [1, 2, 3], inf]

        arr_out6 = core.slice_array(ixs6, arr1, None, False)
        arr_out6_test = [6, 7, 8, 11, 12, 13, 16, 17, 18,
                         31, 32, 33, 36, 37, 38, 41, 42, 43]
        eq6 = (arr_out6.data == arr_out6_test)

        self.assertTrue(eq1)
        self.assertTrue(eq2)
        self.assertTrue(eq3)
        self.assertTrue(eq4)
        self.assertTrue(eq5)
        self.assertTrue(eq6)


unittest.main()
