import unittest

import numpy as np

from dezero import Variable


class Tests_2(unittest.TestCase):
    def test_step_33_1(self):
        def f(x):
            y = x ** 4 - 2 * x **2
            return y
        x = Variable(np.array(2.0))
        y = f(x)
        y.backward(create_graph=True)
        print(x.grad)
        gx= x.grad
        x.cleargrad()
        gx.backward()
        print(x.grad)