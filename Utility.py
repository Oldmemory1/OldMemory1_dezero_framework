import numpy as np
from typing_extensions import override


class Variable:
    def __init__(self,data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self,func):
        self.creator = func

    def backward(self):
        f = self.creator # 获取函数
        if f is not None:
            x = f.input # 获取函数输入
            x.grad = f.backward(self.grad) # 调用函数的backward方法
            x.backward() # 递归调用前一个变量的backward方法

class Function:
    def __call__(self,input:Variable):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self,x):
        raise NotImplementedError()
    def backward(self,grad):
        raise NotImplementedError()

class Square(Function):
    @override
    def forward(self,x):
        return x ** 2
    @override
    def backward(self,gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    @override
    def forward(self,x):
        return np.exp(x)
    @override
    def backward(self,gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def numerical_diff(f,x:Variable,eps = 1e-4):
    x0 = Variable(x.data-eps)
    x1 = Variable(x.data+eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data)/(2*eps)
