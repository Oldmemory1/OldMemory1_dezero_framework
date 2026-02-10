import os.path
import subprocess


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

def get_dot_graph(output, verbose=False):
    txt = ''
    functions = []
    seen_set = set()
    def add_func(func_):
        if func_ not in seen_set:
            functions.append(func_)
            seen_set.add(func_)
    add_func(func_=output.creator)
    txt += _dot_var(output,verbose)
    while functions:
        func = functions.pop()
        txt += _dot_func(func)
        for input_ in func.inputs:
            txt += _dot_var(input_,verbose)
            if input_.creator is not None:
                add_func(func_=input_.creator)
    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    # 输出计算图
    dot_graph = get_dot_graph(output, verbose)
    tmp_dir = os.path.join(os.getcwd(),'tmp')
    if not(os.path.exists(tmp_dir)):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir,'tmp_graph.dot')
    with open(graph_path,'w') as f:
        f.write(dot_graph)
    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path,extension,to_file)
    subprocess.run(cmd, shell=True)
    os.remove(graph_path)