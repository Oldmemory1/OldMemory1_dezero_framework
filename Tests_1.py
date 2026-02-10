import unittest

import numpy as np

if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable

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


