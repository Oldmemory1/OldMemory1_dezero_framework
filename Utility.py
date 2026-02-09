import contextlib
import weakref

import numpy as np
from numpy import ndarray
from typing_extensions import override, Optional

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name,value):
    old_value = getattr(Config,name)
    setattr(Config,name,value)
    try:
        yield
    finally:
        setattr(Config,name,old_value)

class Variable:
    def __init__(self,data:Optional[ndarray],name = None):
        if data is not None:
            if not isinstance(data,ndarray):
                raise TypeError("{} is not supported".format(type(data)))
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self,func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self,retain_grad = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = []
        seen_set = set()

        def add_func(func):
            if func not in seen_set:
                funcs.append(func)
                seen_set.add(func)
                funcs.sort(key=lambda func: func.generation) # generation较大的，先算反向传播

        add_func(func=self.creator)

        while funcs:
            f = funcs.pop() #获取函数
            gys = [output().grad for output in f.outputs] # 获取outputs的导数汇集到列表中
            gxs = f.backward(*gys)
            if not isinstance(gxs,tuple):
                gxs = (gxs,)
            for x,gx in zip(f.inputs,gxs): # f.input[i]的导数对应gxs[i]
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(func=x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # y 是 weakref

    def cleargrad(self): # 清除导数
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self): #维度
        return self.data.ndim

    @property
    def size(self): #元素数量
        return self.data.size

    @property
    def dtype(self): #数据类型
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    @override
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n','\n'+' '*9)
        return 'variable('+p+')'

    def __mul__(self,other):
        return mul(self,other)

    def __add__(self,other):
        return add(self,other)

class Function:
    def __call__(self,*inputs):
        inputs = [as_variable(obj=input_) for input_ in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 解包
        if not isinstance(ys,tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs)>1 else outputs[0]

    def forward(self,*xs):
        raise NotImplementedError()
    def backward(self,*gys):
        raise NotImplementedError()

class Square(Function):
    @override
    def forward(self,x):
        return x ** 2
    @override
    def backward(self,gy):
        x = self.inputs[0].data
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
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx

def exp(x):
    return Exp()(x)

class Add(Function):
    @override
    def forward(self,x0,x1):
        y = x0 + x1
        return y
    @override
    def backward(self,gy):
        return gy,gy
def add(x0,x1):
    x1 = as_array(x1)
    return Add()(x0,x1)

class Mul(Function):
    @override
    def forward(self,x0,x1):
        y = x0 * x1
        return y
    @override
    def backward(self,gy):
        x0 = self.inputs[0].data
        x1 = self.inputs[1].data
        return gy * x1 , gy * x0
def mul(x0,x1):
    return Mul()(x0,x1)

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

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    else:
        return Variable(obj)