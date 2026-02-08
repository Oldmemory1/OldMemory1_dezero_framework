import numpy as np
from numpy import ndarray
from typing_extensions import override,Optional


class Variable:
    def __init__(self,data:Optional[ndarray]):
        if data is not None:
            if not isinstance(data,ndarray):
                raise TypeError("{} is not supported".format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self,func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() #获取函数
            x, y = f.input, f.output #获取函数的输入和输出
            x.grad = f.backward(gy=y.grad) # 反向传播计算输入的梯度
            if x.creator is not None: # 将前一个函数添加到集合中
                funcs.append(x.creator)

class Function:
    def __call__(self,input:Variable):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
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

def square(x):
    return Square()(x)

class Exp(Function):
    @override
    def forward(self,x):
        return np.exp(x)
    @override
    def backward(self,gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def exp(x):
    return Exp()(x)

def numerical_diff(f,x:Variable,eps = 1e-4):
    x0 = Variable(x.data-eps)
    x1 = Variable(x.data+eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data)/(2*eps)

def as_array(x):
    if np.isscalar(x): # x是否为标量
        return np.array(x) # 将其转换为ndarray实例
    return x