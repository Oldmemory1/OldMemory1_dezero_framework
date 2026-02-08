import unittest

import numpy as np

from Utility import Variable


class Tests(unittest.TestCase):
    def test_step01_1(self):
        data = np.array(1.0)
        x = Variable(data)
        print(x.data)