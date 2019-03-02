import unittest

import numpy as np

import core
import linalg


class test_special_matricies(unittest.TestCase):
    def test_diagonal(self):
        arr1 = [[5, 0, 0],
                [0, 6, 0],
                [0, 0, 7]]
        arr1 = core.tomdarray(arr1)
        arr1_diagnoal = linalg.diagonal(arr1)

        arr2_diagonal = core.tomdarray([5, 6, 7])
        arr2 = linalg.diagonal(arr2_diagonal)

        eq1 = all((arr1 == arr2).data)
        eq2 = all((arr1_diagnoal == arr2_diagonal).data)

        self.assertTrue(eq1)
        self.assertTrue(eq2)

    def test_identity(self):
        arr1 = [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]

        arr2 = linalg.identity(3)

        eq = all((arr1 == arr2).data)

        self.assertTrue(eq)
