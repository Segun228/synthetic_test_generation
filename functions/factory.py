from functions.generics import Generic_Function
from functions.function_types import FUNCTION_TYPES
from functions.function_classes import FUNCTION_CLASSES
import warnings
import math


def function_factory(function_type: str, **coefficients) -> Generic_Function:
    if function_type not in FUNCTION_TYPES:
        raise NotImplementedError("Currently we do not support this type of functions")
    expected_args = FUNCTION_TYPES[function_type]

    default_values = {
        'linear': {'k': 1.0, 'b': 0},
        'quadratic': {'a': 1.0, 'b': 0, 'c': 0},
        'cubic': {'a': 1.0, 'b': 0, 'c': 0, 'd': 0},
        'exponential': {'a': 1.0, 'lambda_': 1.0},
        'logarithm': {'k': 1.0, 'a': math.e, 'b': 0},
        'sinus': {'a': 1.0, 'omega': 1.0},
        'cosine': {'a': 1.0, 'omega': 1.0},
        'tangent': {'a': 1.0, 'omega': 1.0},
        'cotangent': {'a': 1.0, 'omega': 1.0}
    }
    for arg_name in expected_args:
        if arg_name not in coefficients:
            if function_type in default_values and arg_name in default_values[function_type]:
                coefficients[arg_name] = default_values[function_type][arg_name]
                warnings.warn(
                    f"Using default value {coefficients[arg_name]} for '{arg_name}' in {function_type} function",
                    UserWarning,
                    stacklevel=2
                )
            else:
                raise ValueError(f"Missing required coefficient '{arg_name}' for {function_type} function")
    func_class = FUNCTION_CLASSES.get(function_type)
    if not func_class:
        raise NotImplementedError("Unfortunately we do not currently support this type of function")
    return func_class(**coefficients)