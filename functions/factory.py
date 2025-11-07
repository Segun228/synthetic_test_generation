from .generics import Generic_Function
from .function_types import FUNCTION_TYPES, FUNCTION_CLASSES

def function_factory(function_type: str, **coefficients) -> Generic_Function:
    if function_type not in FUNCTION_TYPES:
        raise NotImplementedError("Currently we do not support this type of functions")
    expected_args = FUNCTION_TYPES[function_type]
    for arg_name in expected_args:
        if arg_name not in coefficients:
            raise ValueError(f"Missing required coefficient '{arg_name}' for {function_type} function")
    func_class = FUNCTION_CLASSES.get(function_type)
    if not func_class:
        raise NotImplementedError("Unfortunately we do not currently support this type of function")
    return func_class(coefficients)