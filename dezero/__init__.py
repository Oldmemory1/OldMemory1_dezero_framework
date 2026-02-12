simple_core = False

if simple_core:
    from dezero.simple_core import Variable, Function, Square, Exp, numerical_diff, square, exp, Add, add, mul, no_grad
else:
    from dezero.core import Variable, Function, Square, Exp, Add, add, mul, Mul, using_config, no_grad