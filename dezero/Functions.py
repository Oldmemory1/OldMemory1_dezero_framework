import numpy as np
from typing_extensions import override

from dezero.core import Function


class Sin(Function):
    @override
    def forward(self, x):
        y = np.sin(x)
        return y
    @override
    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):
    @override
    def forward(self, x):
        y = np.cos(x)
        return y
    @override
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)

class Tanh(Function):
    @override
    def forward(self, x):
        y = np.tanh(x)
        return y
    @override
    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx
def tanh(x):
    return Tanh()(x)
