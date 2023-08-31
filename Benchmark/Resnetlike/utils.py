import os
import distutils.dir_util
import pprint, ast

def load_py_module_from_path(module_path, module_name=None):
    if module_path.find(':') > 0:
        split_path = module_path.split(':')
        module_path = split_path[0]
        function_name = split_path[1]
    else:
        function_name = None

    if module_name is None:
        module_name = module_path.replace('/', '_').replace('.', '_')

    assert os.path.isfile(module_path)

    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    any_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(any_module)
    if function_name is None:
        return any_module
    else:
        return getattr(any_module, function_name)

def mkfilepath(filename):
    distutils.dir_util.mkpath(os.path.dirname(filename))


def mkdir(dirname):
    distutils.dir_util.mkpath(dirname)


def smart_round(x, base=None):
    if base is None:
        if x > 32 * 8:
            round_base = 32
        elif x > 16 * 8:
            round_base = 16
        else:
            round_base = 8
    else:
        round_base = base

    return max(round_base, round(x / float(round_base)) * round_base)

def save_pyobj(filename, pyobj):
    mkfilepath(filename)
    the_s = pprint.pformat(pyobj, indent=2, width=120, compact=True)
    with open(filename, 'w') as fid:
        fid.write(the_s)


def load_pyobj(filename):
    with open(filename, 'r') as fid:
        the_s = fid.readlines()

    if isinstance(the_s, list):
        the_s = ''.join(the_s)

    the_s = the_s.replace('inf', '1e20')
    pyobj = ast.literal_eval(the_s)
    return pyobj

