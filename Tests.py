import unittest

import numpy as np

from Utility import Variable, Function, Square, Exp, numerical_diff, square, exp, Add, add, mul


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
    def test_step11_1(self):
        xs = [Variable(np.array(2)),Variable(np.array(3))]
        f = Add()
        ys = f(xs)
        y = ys[0]
        print(y.data)
    def test_step12_1(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        f = Add()
        y = f(x0, x1)
        print(y.data)
    def test_step12_2(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y = add(x0, x1)
        print(y.data)
    def test_step13_1(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        z = add(square(x), square(y))
        z.backward()
        print(z.data)
        print(x.grad)
        print(y.grad)
    def test_step14_1(self):
        x = Variable(np.array(3.0))
        y = add(x,x)
        z = add(y,x)
        z.backward()
        print(x.grad)
    def test_step14_2(self):
        x = Variable(np.array(3.0))
        y = add(x,x)
        y.backward()
        print(x.grad)

        x.cleargrad()
        y = add(add(x,x),x)
        y.backward()
        print(x.grad)
    def test_step16_1(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a)) # y = (x^2)^2 + (x^2)^2
        y.backward()
        print(y.data)
        print(x.grad)
    def test_step18_1(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        t = add(x0, x1)
        y = add(x0, t)
        y.backward()
        print(y.grad,t.grad)
        print(x0.grad,x1.grad)
    def test_step19_1(self):
        x = Variable(np.array([[1,2,3],[4,5,6]]))
        print(len(x))
    def test_step20_1(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        c = Variable(np.array(1.0))

        y = add(mul(a,b),c)
        y.backward()
        print(y)
        print(a.grad)
        print(b.grad)
    def test_step20_2(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        y = a * b
        print(y)

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data,expected)
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad,expected)
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square,x)
        flg = np.allclose(x.grad,num_grad)
        self.assertTrue(flg)
