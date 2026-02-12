import math
import unittest
from typing_extensions import override

import numpy as np

from dezero.utils import _dot_var, _dot_func, plot_dot_graph

if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable, Function


def sphere(x,y):
    z = x**2 + y**2
    return z
def matyas(x,y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z
def goldstein(x, y):
    z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
                30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return z
def rosenbrock(x0,x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

class Sin(Function):
    @override
    def forward(self, x):
        y = np.sin(x)
        return y
    @override
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

def mysinx(x, threshold = 0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2*i+1)
        t = c * x ** (2*i+1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

class Tests_1(unittest.TestCase):
    def test_step_23_1(self):
        x = Variable(np.array(1.0))
        y = (x + 3) ** 2
        y.backward()
        print(y)
        print(x.grad)
    def test_step_24_1(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = sphere(x, y)
        z.backward()
        print(x.grad,y.grad)
    def test_step_24_2(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = matyas(x, y)
        z.backward()
        print(x.grad,y.grad)
    def test_step_24_3(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = goldstein(x, y)
        print(z)
        z.backward()
        print(x.grad,y.grad)
    def test_step_26_1(self):
        x = Variable(np.random.randn(2,3))
        x.name = 'x'
        print(_dot_var(x))
        print(_dot_var(x, verbose=True))
    def test_step_26_2(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        y = x0 + x1
        txt = _dot_func(y.creator)
        print(txt)
    def test_step_26_3(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = goldstein(x,y)
        z.backward()
        x.name = 'x'
        y.name = 'y'
        z.name = 'z'
        plot_dot_graph(z,verbose=False,to_file='graph.png')
    def test_step_27_1(self):
        x = Variable(np.array(np.pi/4))
        y = sin(x)
        y.backward()
        print(y.data)
        print(x.grad)
    def test_step_27_2(self):
        x = Variable(np.array(np.pi/4))
        y = mysinx(x)
        y.backward()
        print(y.data)
        print(x.grad)
        x.name = 'x'
        y.name = 'y'
        plot_dot_graph(y,verbose=False,to_file='graph_mysin.png')
    def test_step_28_1(self):
        x0 = Variable(np.array(0.0))
        x1 = Variable(np.array(2.0))
        y = rosenbrock(x0,x1)
        y.backward()
        print(y.data)
        print(x0.grad)
        print(x1.grad)
    def test_step_28_2(self):
        x0 = Variable(np.array(0.0))
        x1 = Variable(np.array(2.0))
        learning_rate = 0.001
        iterations = 1000
        for i in range(iterations):
            print(x0,x1)
            y = rosenbrock(x0,x1)
            x0.cleargrad()
            x1.cleargrad()
            y.backward()
            x0.data -= learning_rate * x0.grad
            x1.data -= learning_rate * x1.grad
    def test_step_29_1(self):
        def f(x):
            return x ** 4 - 2 * x ** 2
        def g2(x):
            return 12 * x **2 - 4
        x = Variable(np.array(2.0))
        iterations = 10
        for i in range(iterations):
            print(i,x)
            y = f(x)
            x.cleargrad()
            y.backward()
            x.data -= x.grad / g2(x.data)
