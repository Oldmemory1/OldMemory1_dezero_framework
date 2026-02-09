import unittest

import numpy as np

from Utility import Variable, Function, Square, Exp, numerical_diff, square, exp


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
    def test_step_06_1(self):
        A = Square()
        B = Exp()
        C = Square()
        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)
        y.grad = np.array(1.0)
        b.grad = C.backward(gy=y.grad)
        a.grad = B.backward(gy=b.grad)
        x.grad = A.backward(gy=a.grad)
        print(x.grad)
    def test_step_07_1(self):
        A = Square()
        B = Exp()
        C = Square()
        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)
        assert y.creator == C
        assert y.creator.input == b
        assert y.creator.input.creator == B
        assert y.creator.input.creator.input == a
        assert y.creator.input.creator.input.creator == A
        assert y.creator.input.creator.input.creator.input == x
        y.grad = np.array(1.0)
        b = C.input
        b.grad = C.backward(gy=y.grad)
        B = b.creator
        a = B.input
        a.grad = B.backward(gy=b.grad)
        A = a.creator
        x = A.input
        x.grad = A.backward(gy=a.grad)
        print(x.grad)
    def test_step_07_2(self):
        # 在step8中也适用
        A = Square()
        B = Exp()
        C = Square()
        x = Variable(np.array(0.5))
        a = A(x)
        b = B(a)
        y = C(b)
        y.grad = np.array(1.0)
        y.backward()
        print(x.grad)
    def test_step09_1(self):
        x = Variable(np.array(0.5))
        a = square(x)
        b = exp(a)
        y = square(b)
        y.grad = np.array(1.0)
        y.backward()
        print(x.grad)
    def test_step09_2(self):
        x = Variable(np.array(0.5))
        y = square(exp(square(x)))
        y.backward()
        print(x.grad)
    def test_step09_3(self):
        try:
            x = Variable(np.array(0.5))
        except TypeError:
            print("Not support np.ndarray")
        try:
            x = Variable(None)
        except TypeError:
            print("Not support None")
        try:
            x = Variable(1.0)
        except TypeError:
            print("Not support float")
    def test_step09_4(self):
        x = np.array([1.0])
        y = x ** 2
        print(type(x),x.ndim)
        print(type(y))
    def test_step09_5(self):
        x = np.array(1.0)
        y = x ** 2
        print(type(x),x.ndim)
        print(type(y)) # 出现numpy.float64
    def test_step09_6(self):
        print(np.isscalar(np.float64(1.0)))
        print(np.isscalar(2.0))
        print(np.isscalar(np.array(1.0)))
        print(np.isscalar(np.array([1.0, 2.0, 3.0])))

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data,expected)
