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


