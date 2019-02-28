import unittest

import numpy as np

import core
import mdarray as md
from core import inf, nan
from mditer2 import (MultiArrayIter, broadcast_nary, broadcast_toshape_iter,
                     concatenate_iter)


class test_mditer_repeat(unittest.TestCase):
    def test_10d_repeat(self):
        arr1 = core.irange([2, 2, 1, 2, 2,
                            2, 2, 1, 2, 2])
        repeats1 = [1, 2, 1, 3, 1,
                    1, 3, 1, 2, 1]

        mditer1 = MultiArrayIter(arr1)

        mditer1.repeats = repeats1

        raxes = [i for i in range(arr1.mdim)]
        base_arr1 = core.repeat(arr1, raxes, repeats1)

        test_arr1 = core.zeros(mditer1.shape)
        test_arr1.data = [i.index for i in mditer1]

        eq = (base_arr1.data == test_arr1.data)

        self.assertTrue(eq)

    def test_5_3d_repeat(self):
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

        mditer1 = MultiArrayIter(arr1)
        mditer2 = MultiArrayIter(arr2)
        mditer3 = MultiArrayIter(arr3)
        mditer4 = MultiArrayIter(arr4)
        mditer5 = MultiArrayIter(arr5)

        mditer1.repeats = repeats1
        mditer2.repeats = repeats2
        mditer3.repeats = repeats3
        mditer4.repeats = repeats4
        mditer5.repeats = repeats5

        raxes = [i for i in range(arr1.mdim)]
        base_arr1 = core.repeat(arr1, raxes, repeats1)
        base_arr2 = core.repeat(arr2, raxes, repeats2)
        base_arr3 = core.repeat(arr3, raxes, repeats3)
        base_arr4 = core.repeat(arr4, raxes, repeats4)
        base_arr5 = core.repeat(arr5, raxes, repeats5)

        test_arr1 = core.zeros(mditer1.shape)
        test_arr1.data = [i.index for i in mditer1]
        test_arr2 = core.zeros(mditer2.shape)
        test_arr2.data = [i.index for i in mditer2]
        test_arr3 = core.zeros(mditer3.shape)
        test_arr3.data = [i.index for i in mditer3]
        test_arr4 = core.zeros(mditer4.shape)
        test_arr4.data = [i.index for i in mditer4]
        test_arr5 = core.zeros(mditer5.shape)
        test_arr5.data = [i.index for i in mditer5]

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

        base_arr1 = core.concatenate(arr1, arr2, arr3, caxis=caxis)
        test_arr1 = concatenate_iter(arr1, arr2, arr3, caxis=caxis)

        eq1 = (base_arr1.data == test_arr1.data)
        self.assertTrue(eq1)

    def test_3_3d_concatenate(self):
        arr1 = core.irange([1, 7, 11])
        arr2 = core.irange([1, 12, 11])
        arr3 = core.irange([1, 1, 11])
        caxis = 1

        base_arr1 = core.concatenate(arr1, arr2, arr3, caxis=caxis)
        test_arr1 = concatenate_iter(arr1, arr2, arr3, caxis=caxis)

        eq1 = (base_arr1.data == test_arr1.data)
        self.assertTrue(eq1)


unittest.main()
