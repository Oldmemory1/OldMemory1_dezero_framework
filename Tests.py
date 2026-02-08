import unittest

import numpy as np

from Utility import Variable, Function, Square


class Tests(unittest.TestCase):
    def test_step01_1(self):
        data = np.array(1.0)
        x = Variable(data)
        print(x.data)
        x.data = np.array(2.0)
        print(x.data)
    def test_step_02_1(self):
        x = Variable(np.array(10))
        f = Function()
        y = f(x)
        print(type(y))
        print(y.data)
    def test_step_02_2(self):
        x = Variable(np.array(10))
        f = Square()
        y = f(x)
        print(type(y))
        print(y.data)