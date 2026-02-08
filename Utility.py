import numpy as np
from typing_extensions import override


class Variable:
    def __init__(self,data):
        self.data = data

class Function:
    def __call__(self,input:Variable):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self,x):
        raise NotImplementedError()

class Square(Function):
    @override
    def forward(self,x):
        return x ** 2

class Exp(Function):
    @override
    def forward(self,x):
        return np.exp(x)