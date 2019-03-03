import unittest

import numpy as np

import core
from core import inf, nan
import core.depreciated as depreciated


class test_mditer_formatting(unittest.TestCase):
    def test_20d_format(self):
        shape1 = [1, 2, 3, 4, 5,
                  1, 1, 1, 1, 1,
                  5, 4, 3, 2, 1,
                  1, 1, 1, 1, 1]

        shape1 = [3, 3, 1, 1, 1, 2]
        arr1 = core.irange(shape1)
        sep1 = " , "

        base_string1 = depreciated.print_array(arr1, sep1, None)
        print(base_string1)

        test_string1 = core.print_array(arr1, sep1, None)
        print(test_string1)

        eq1 = (base_string1 == test_string1)
        self.assertTrue(eq1)


class test_mditer_broadcast(unittest.TestCase):
    def test_3_5d_broadcast(self):
        arr1 = core.irange([5, 2, 1, 1, 1])
        arr2 = core.irange([1, 1, 5, 1, 1])
        arr3 = core.irange([1, 1, 1, 2, 5])

        base_arrs = depreciated.broadcast_arrays(arr1, arr2, arr3)
        base_arr1 = base_arrs[0]
        base_arr2 = base_arrs[1]
        base_arr3 = base_arrs[2]

        test_arrs = core.broadcast(arr1, arr2, arr3)
        test_arr1 = test_arrs[0]
        test_arr2 = test_arrs[1]
        test_arr3 = test_arrs[2]

        eq1 = (base_arr1.data == test_arr1.data)
        eq2 = (base_arr2.data == test_arr2.data)
        eq3 = (base_arr3.data == test_arr3.data)

        self.assertTrue(eq1)
        self.assertTrue(eq2)
        self.assertTrue(eq3)

    def test_4_4d_meshgrid(self):
        grid_components = (range(4), range(2), range(1), range(10))
        base_arrs = depreciated.meshgrid(*grid_components)

        base_arr1 = base_arrs[0]
        base_arr2 = base_arrs[1]
        base_arr3 = base_arrs[2]
        base_arr4 = base_arrs[3]

        test_arrs = core.meshgrid(*grid_components)

        test_arr1 = test_arrs[0]
        test_arr2 = test_arrs[1]
        test_arr3 = test_arrs[2]
        test_arr4 = test_arrs[3]

        eq1 = (base_arr1.data == test_arr1.data)
        eq2 = (base_arr2.data == test_arr2.data)
        eq3 = (base_arr3.data == test_arr3.data)
        eq4 = (base_arr4.data == test_arr4.data)

        self.assertTrue(eq1)
        self.assertTrue(eq2)
        self.assertTrue(eq3)
        self.assertTrue(eq4)


class test_mditer_repeat(unittest.TestCase):
    def test_10d_repeat(self):
        arr1 = core.irange([2, 2, 1, 2, 2,
                            2, 2, 1, 2, 2])
        repeats1 = [1, 2, 1, 3, 1,
                    1, 3, 1, 2, 1]

        raxes = [i for i in range(arr1.mdim)]

        base_arr1 = depreciated.repeat(arr1, raxes, repeats1)
        test_arr1 = core.repeat(arr1, raxes, repeats1)

        eq = (base_arr1.data == test_arr1.data)
        self.assertTrue(eq)

    def test_3_5d(self):
        arr1 = core.irange([1, 1, 1, 1, 5])
        arr2 = core.irange([1, 1, 4, 1, 1])
        arr3 = core.irange([5, 1, 1, 2, 1])
        arr4 = core.irange([1, 2, 1, 2, 1])
        arr5 = core.irange([1, 5, 2, 2, 10])

        repeats1 = [3, 3, 3, 3, 1]
        repeats2 = [2, 3, 1, 2, 1]
        repeats3 = [2, 2, 2, 2, 2]
        repeats4 = [2, 1, 2, 1, 2]
        repeats5 = [1, 1, 1, 1, 3]

        raxes = [i for i in range(arr1.mdim)]
        base_arr1 = depreciated.repeat(arr1, raxes, repeats1)
        base_arr2 = depreciated.repeat(arr2, raxes, repeats2)
        base_arr3 = depreciated.repeat(arr3, raxes, repeats3)
        base_arr4 = depreciated.repeat(arr4, raxes, repeats4)
        base_arr5 = depreciated.repeat(arr5, raxes, repeats5)

        test_arr1 = core.repeat(arr1, raxes, repeats1)
        test_arr2 = core.repeat(arr2, raxes, repeats2)
        test_arr3 = core.repeat(arr3, raxes, repeats3)
        test_arr4 = core.repeat(arr4, raxes, repeats4)
        test_arr5 = core.repeat(arr5, raxes, repeats5)

        eq1 = (base_arr1.data == test_arr1.data)
        eq2 = (base_arr2.data == test_arr2.data)
        eq3 = (base_arr3.data == test_arr3.data)
        eq4 = (base_arr4.data == test_arr4.data)
        eq5 = (base_arr5.data == test_arr5.data)

        self.assertTrue(eq1)
        self.assertTrue(eq2)
        self.assertTrue(eq3)
        self.assertTrue(eq4)
        self.assertTrue(eq5)

    def test_5_3d2_repeat(self):
        arr1 = core.irange([5, 1, 2])
        arr2 = core.irange([3, 2, 7])
        arr3 = core.irange([4, 1, 1])
        arr4 = core.irange([1, 4, 1])
        arr5 = core.irange([1, 1, 4])

        repeats1 = [2, 2, 2]
        repeats2 = [1, 1, 2]
        repeats3 = [1, 10, 2]
        repeats4 = [1, 2, 3]
        repeats5 = [10, 10, 2]

        raxes = [i for i in range(arr1.mdim)]
        base_arr1 = depreciated.repeat(arr1, raxes, repeats1)
        base_arr2 = depreciated.repeat(arr2, raxes, repeats2)
        base_arr3 = depreciated.repeat(arr3, raxes, repeats3)
        base_arr4 = depreciated.repeat(arr4, raxes, repeats4)
        base_arr5 = depreciated.repeat(arr5, raxes, repeats5)

        test_arr1 = core.repeat(arr1, raxes, repeats1)
        test_arr2 = core.repeat(arr2, raxes, repeats2)
        test_arr3 = core.repeat(arr3, raxes, repeats3)
        test_arr4 = core.repeat(arr4, raxes, repeats4)
        test_arr5 = core.repeat(arr5, raxes, repeats5)

        eq1 = (base_arr1.data == test_arr1.data)
        eq2 = (base_arr2.data == test_arr2.data)
        eq3 = (base_arr3.data == test_arr3.data)
        eq4 = (base_arr4.data == test_arr4.data)
        eq5 = (base_arr5.data == test_arr5.data)

        self.assertTrue(eq1)
        self.assertTrue(eq2)
        self.assertTrue(eq3)
        self.assertTrue(eq4)
        self.assertTrue(eq5)


class test_mditer_concatenate(unittest.TestCase):
    def test_3_5d_concatenate(self):
        arr1 = core.irange([5, 2, 5, 2, 1])
        arr2 = core.irange([5, 2, 1, 2, 1])
        arr3 = core.irange([5, 2, 3, 2, 1])
        caxis = 2

        base_arr1 = depreciated.concatenate(arr1, arr2, arr3, caxis=caxis)
        test_arr1 = core.concatenate(arr1, arr2, arr3, caxis=caxis)

        eq1 = (base_arr1.data == test_arr1.data)
        self.assertTrue(eq1)

    def test_3_3d_concatenate(self):
        arr1 = core.irange([1, 7, 11])
        arr2 = core.irange([1, 12, 11])
        arr3 = core.irange([1, 1, 11])
        caxis = 1

        base_arr1 = depreciated.concatenate(arr1, arr2, arr3, caxis=caxis)
        test_arr1 = core.concatenate(arr1, arr2, arr3, caxis=caxis)

        eq1 = (base_arr1.data == test_arr1.data)
        self.assertTrue(eq1)


unittest.main()
