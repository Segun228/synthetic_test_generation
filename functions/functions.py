from .generics import Generic_Function
from typing import Any
import math

FUNCTION_TYPES = {
    "linear":{
        "k":float,
        "b":float
    },
    "quadratic":{
        "a":float,
        "b":float,
        "c":float,
    },
    "cubic":{
        "a":float,
        "b":float,
        "c":float,
        "d":float,
    },
    "exponential":{
        "a":float,
        "lambda_":float
    },
    "logarithm":{
        "k":float,
        "a":int,
        "b":int
    },
    "sinus":{
        "a":float,
        "omega":float
    },
    "cosine":{
        "a":float,
        "omega":float
    },
    "tangent":{
        "a":float,
        "omega":float
    },
    "cotangent":{
        "a":float,
        "omega":float
    }
}


class LinearFunction(Generic_Function):
    def __init__(self, k: float, b: float, **kwargs):
        super().__init__('linear', k=k, b=b, **kwargs)

    def _validate_coefficients(self, coefficients: dict) -> None:
        required = {'k', 'b'}
        if not required.issubset(coefficients.keys()):
            missing = required - set(coefficients.keys())
            raise ValueError(f"Missing required coefficients: {missing}")
        for coef in ['k', 'b']:
            if coef in coefficients and not isinstance(coefficients[coef], (int, float)):
                raise TypeError(f"Coefficient '{coef}' must be numeric")

    def _evaluate(self, x: Any) -> Any:
        return self.k * x + self.b



class QuadraticFunction(Generic_Function):
    def __init__(
            self, 
            a: float, 
            b: float, 
            c: float,
            **kwargs
        ):
        super().__init__(
            'quadratic', 
            a=a, 
            b=b, 
            c=c,
            **kwargs
        )

    def _validate_coefficients(self, coefficients: dict) -> None:
        required = {'a', 'b', 'c'}
        if not required.issubset(coefficients.keys()):
            missing = required - set(coefficients.keys())
            raise ValueError(f"Missing required coefficients: {missing}")
        for coef in ['a', 'b', 'c']:
            if coef in coefficients and not isinstance(coefficients[coef], (int, float)):
                raise TypeError(f"Coefficient '{coef}' must be numeric")

    def _evaluate(self, x: Any) -> Any:
        return self.a * x**2 + self.b * x + self.c




class CubicFunction(Generic_Function):
    def __init__(self, a: float, b: float, c: float, d: float, **kwargs):
        super().__init__('cubic', a=a, b=b, c=c, d=d, **kwargs)

    def _validate_coefficients(self, coefficients: dict) -> None:
        required = {'a', 'b', 'c', 'd'}
        if not required.issubset(coefficients.keys()):
            missing = required - set(coefficients.keys())
            raise ValueError(f"Missing required coefficients: {missing}")
        for coef in ['a', 'b', 'c', 'd']:
            if coef in coefficients and not isinstance(coefficients[coef], (int, float)):
                raise TypeError(f"Coefficient '{coef}' must be numeric")

    def _evaluate(self, x: Any) -> Any:
        return self.a * x**3 + self.b * x**2 + self.c * x + self.d


class ExponentialFunction(Generic_Function):
    def __init__(self, a: float, lambda_: float, **kwargs):
        super().__init__('exponential', a=a, lambda_=lambda_, **kwargs)

    def _validate_coefficients(self, coefficients: dict) -> None:
        required = {'a', 'lambda_'}
        if not required.issubset(coefficients.keys()):
            missing = required - set(coefficients.keys())
            raise ValueError(f"Missing required coefficients: {missing}")
        for coef in ['a', 'lambda_']:
            if coef in coefficients and not isinstance(coefficients[coef], (int, float)):
                raise TypeError(f"Coefficient '{coef}' must be numeric")

    def _evaluate(self, x: Any) -> Any:
        return self.a * math.exp(self.lambda_ * x)


class LogarithmFunction(Generic_Function):
    def __init__(self, k: float, a: int, b: int, **kwargs):
        super().__init__('logarithm', k=k, a=a, b=b, **kwargs)

    def _validate_coefficients(self, coefficients: dict) -> None:
        required = {'k', 'a', 'b'}
        if not required.issubset(coefficients.keys()):
            missing = required - set(coefficients.keys())
            raise ValueError(f"Missing required coefficients: {missing}")
        if 'k' in coefficients and not isinstance(coefficients['k'], (int, float)):
            raise TypeError("Coefficient 'k' must be numeric")
        if 'a' in coefficients and not (isinstance(coefficients['a'], (int, float)) and coefficients['a'] > 0 ):
            raise TypeError("Coefficient 'a' must be integer")
        if 'b' in coefficients and not isinstance(coefficients['b'], int):
            raise TypeError("Coefficient 'b' must be integer")
        if 'a' in coefficients and coefficients['a'] <= 0:
            raise ValueError("Base 'a' must be positive")
        if 'a' in coefficients and coefficients['a'] == 1:
            raise ValueError("Base 'a' cannot be 1")

    def _evaluate(self, x: Any) -> Any:
        if x <= 0:
            raise ValueError("Logarithm is defined only for positive numbers")
        return self.k * math.log(x + self.b, self.a)


class SinusFunction(Generic_Function):
    def __init__(self, a: float, omega: float, **kwargs):
        super().__init__('sinus', a=a, omega=omega, **kwargs)

    def _validate_coefficients(self, coefficients: dict) -> None:
        required = {'a', 'omega'}
        if not required.issubset(coefficients.keys()):
            missing = required - set(coefficients.keys())
            raise ValueError(f"Missing required coefficients: {missing}")
        for coef in ['a', 'omega']:
            if coef in coefficients and not isinstance(coefficients[coef], (int, float)):
                raise TypeError(f"Coefficient '{coef}' must be numeric")

    def _evaluate(self, x: Any) -> Any:
        return self.a * math.sin(self.omega * x)


class CosineFunction(Generic_Function):
    def __init__(self, a: float, omega: float, **kwargs):
        super().__init__('cosine', a=a, omega=omega, **kwargs)

    def _validate_coefficients(self, coefficients: dict) -> None:
        required = {'a', 'omega'}
        if not required.issubset(coefficients.keys()):
            missing = required - set(coefficients.keys())
            raise ValueError(f"Missing required coefficients: {missing}")
        for coef in ['a', 'omega']:
            if coef in coefficients and not isinstance(coefficients[coef], (int, float)):
                raise TypeError(f"Coefficient '{coef}' must be numeric")

    def _evaluate(self, x: Any) -> Any:
        return self.a * math.cos(self.omega * x)


class TangentFunction(Generic_Function):
    def __init__(self, a: float, omega: float, **kwargs):
        super().__init__('tangent', a=a, omega=omega, **kwargs)

    def _validate_coefficients(self, coefficients: dict) -> None:
        required = {'a', 'omega'}
        if not required.issubset(coefficients.keys()):
            missing = required - set(coefficients.keys())
            raise ValueError(f"Missing required coefficients: {missing}")
        for coef in ['a', 'omega']:
            if coef in coefficients and not isinstance(coefficients[coef], (int, float)):
                raise TypeError(f"Coefficient '{coef}' must be numeric")

    def _evaluate(self, x: Any) -> Any:
        result = self.a * math.tan(self.omega * x)
        if abs(result) > 1e10:
            raise ValueError(f"Tangent is undefined for x = {x}")
        return result


class CotangentFunction(Generic_Function):
    def __init__(self, a: float, omega: float, **kwargs):
        super().__init__('cotangent', a=a, omega=omega, **kwargs)

    def _validate_coefficients(self, coefficients: dict) -> None:
        required = {'a', 'omega'}
        if not required.issubset(coefficients.keys()):
            missing = required - set(coefficients.keys())
            raise ValueError(f"Missing required coefficients: {missing}")
        for coef in ['a', 'omega']:
            if coef in coefficients and not isinstance(coefficients[coef], (int, float)):
                raise TypeError(f"Coefficient '{coef}' must be numeric")

    def _evaluate(self, x: Any) -> Any:
        tan_value = math.tan(self.omega * x)
        if abs(tan_value) < 1e-10: 
            raise ValueError(f"Cotangent is undefined for x = {x}")
        return self.a / tan_value