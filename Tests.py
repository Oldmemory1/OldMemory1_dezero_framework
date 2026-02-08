import unittest

import numpy as np

from Utility import Variable, Function, Square, Exp, numerical_diff


class Tests(unittest.TestCase):
    def test_step_01_1(self):
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
    def test_step_03_1(self):
        A = Square()
        B = Exp()
        C = Square()
        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)
        print(y.data)
    def test_step_04_1(self):
        f = Square()
        x = Variable(np.array(2.0))
        dy = numerical_diff(f,x)
        print(dy)
    def test_step_04_2(self):
        def f(x):
            A = Square()
            B = Exp()
            C = Square()
            return C(B(A(x)))
        x = Variable(np.array(0.5))
        dy = numerical_diff(f,x)
        print(dy)