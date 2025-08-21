import inspect

def get_decorated_functions(module, ignore=("main",)):
    """
    Returns a list of functions decorated with `function_metadata` in a module.
    Excludes functions by name (like 'main', 'helper', etc.).
    """
    functions = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if hasattr(func, "meta") and name not in ignore:
            functions.append((name, func))
    return functions


