def _dot_var(v , verbose = False):
    #  Variable类转为dot语言
    dot_var = '{} [label="{}" , color = orange , style = filled]\n'
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.name)
    return dot_var.format(id(v), name)

def _dot_func(f):
    # Function类转为dot语言
    dot_func = '{} [label="{}", color = lightgrey , style = filled, shape = box]\n'
    txt = dot_func.format(id(f),f.__class__.__name__)
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x),id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f),id(y())) #outputs为weakref
    return txt