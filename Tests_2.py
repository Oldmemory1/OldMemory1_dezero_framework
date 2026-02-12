import unittest

import numpy as np
from matplotlib import pyplot as plt

import dezero.Functions as F
from dezero import Variable
from dezero.utils import plot_dot_graph


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
    def test_step_33_2(self):
        def f(x):
            y = x ** 4 - 2 * x **2
            return y
        x = Variable(np.array(2.0))
        iterations = 10
        for i in range(iterations):
            print(i,x)
            y = f(x)
            x.cleargrad()
            y.backward(create_graph=True)
            gx = x.grad
            x.cleargrad()
            gx.backward()
            gx2 = x.grad
            x.data -= gx.data / gx2.data
    def test_step_34_1(self):
        x =Variable(np.array(1.0))
        y = F.sin(x)
        y.backward(create_graph=True)
        for i in range(3):
            gx = x.grad
            x.cleargrad()
            gx.backward(create_graph=True)
            print(x.grad)
    def test_step_34_2(self):
        x = Variable(np.linspace(-7,7,200))
        y = F.sin(x)
        y.backward(create_graph=True)
        logs = [y.data]
        for i in range(3):
            logs.append(x.grad.data)
            gx = x.grad
            x.cleargrad()
            gx.backward(create_graph=True)
        labels = ["y=sin(x)","y'","y''","y'''"]
        for i,v in enumerate(logs):
            plt.plot(x.data,logs[i],label=labels[i])
        plt.legend(loc = 'lower right')
        plt.show()
    def test_step_35_1(self):
        x = Variable(np.array(1.0))
        y = F.tanh(x)
        x.name = 'x'
        y.name = 'y'
        y.backward(create_graph=True)
        iterations = 0
        for i in range(iterations):
            gx = x.grad
            x.cleargrad()
            gx.backward(create_graph=True)
        gx = x.grad
        gx.name = 'gx' + str(iterations+1)
        plot_dot_graph(gx,verbose=False,to_file='tanh.png')
    def test_step_36_1(self):
        x = Variable(np.array(2.0))
        y = x ** 2
        y.backward(create_graph=True)
        gx = x.grad
        x.cleargrad()
        z = gx ** 3 + y
        z.backward()
        print(x.grad)