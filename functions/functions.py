from .generics import Generic_Function
from typing import Any
import math
from functions.function_types import FUNCTION_TYPES


class LinearFunction(Generic_Function):
    def __init__(self, k: float = 0.5, b: float = 0, **kwargs):
        super().__init__('linear', k=k, b=b, **kwargs)

    def _validate_coefficients(self, coefficients: dict) -> None:
        required = {'k', 'b'}
        for coef_name in coefficients.keys():
            if coef_name not in required:
                raise ValueError(f"Unknown coefficient: {coef_name}")
            if not isinstance(coefficients[coef_name], (int, float)):
                raise TypeError(f"Coefficient '{coef_name}' must be numeric")

    def _evaluate(self, x: Any) -> Any:
        return self.k * x + self.b

    def inverse_function(self, argument: Any) -> Any:
        return (argument - self.b) / self.k



class QuadraticFunction(Generic_Function):
    def __init__(
            self, 
            a: float = 2, 
            b: float = -3, 
            c: float = 2,
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

def inverse_function(self, y: Any) -> Any:
    discriminant = self.b**2 - 4*self.a*(self.c - y)
    if discriminant < 0:
        raise ValueError("No real solution")
    return (-self.b + math.sqrt(discriminant)) / (2 * self.a)




class CubicFunction(Generic_Function):
    def __init__(self, a: float = 2, b: float = -4, c: float = 6, d: float = 12, **kwargs):
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

    def inverse_function(self, y: Any) -> Any:
        from scipy.optimize import fsolve
        def equation(x):
            return self.a*x**3 + self.b*x**2 + self.c*x + self.d - y
        return fsolve(equation, 0)[0]


class ExponentialFunction(Generic_Function):
    def __init__(self, a: float = 1, lambda_: float = 1, **kwargs):
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
    def __init__(self, k: float = 1, a: int = 2, b: int = 1, **kwargs):
        super().__init__('logarithm', k=k, a=a, b=b, **kwargs)

    def _validate_coefficients(self, coefficients: dict) -> None:
        required = {'k', 'a', 'b'}
        if not required.issubset(coefficients.keys()):
            missing = required - set(coefficients.keys())
            raise ValueError(f"Missing required coefficients: {missing}")
        if 'k' in coefficients and not isinstance(coefficients['k'], (int, float)):
            raise TypeError("Coefficient 'k' must be numeric")
        if 'a' in coefficients and not (isinstance(coefficients['a'], (int, float))):
            raise TypeError("Coefficient 'a' must be integer")
        if 'b' in coefficients and not isinstance(coefficients['b'], (int, float)):
            raise TypeError("Coefficient 'b' must be integer")
        if 'a' in coefficients and coefficients['a'] <= 0:
            raise ValueError("Base 'a' must be positive")
        if 'a' in coefficients and coefficients['a'] == 1:
            raise ValueError("Base 'a' cannot be 1")

    def _evaluate(self, x: Any) -> Any:
        if x <= 0:
            raise ValueError("Logarithm is defined only for positive numbers")
        return self.k * math.log(x + self.b, self.a)

    def inverse_function(self, y: Any) -> Any:
        if y <= 0:
            raise ValueError("Exponential output must be positive")
        return math.log(y / self.a) / self.lambda_


class SinusFunction(Generic_Function):
    def __init__(self, a: float = 1, omega: float = 1, **kwargs):
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

    def inverse_function(self, y: Any) -> Any:
        if abs(y) > abs(self.a):
            raise ValueError("Value outside function range")
        return math.asin(y / self.a) / self.omega


class CosineFunction(Generic_Function):
    def __init__(self, a: float = 1, omega: float = 1, **kwargs):
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

    def inverse_function(self, y: Any) -> Any:
        if abs(y) > abs(self.a):
            raise ValueError("Value outside function range")
        return math.acos(y / self.a) / self.omega

class TangentFunction(Generic_Function):
    def __init__(self, a: float = 1, omega: float = 1, **kwargs):
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

    def inverse_function(self, y: Any) -> Any:
        return math.atan(y / self.a) / self.omega


class CotangentFunction(Generic_Function):
    def __init__(self, a: float = 1, omega: float = 1, **kwargs):
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

    def inverse_function(self, y: Any) -> Any:
        if y == 0:
            raise ValueError("Cotangent inverse undefined for 0")
        return math.atan(self.a / y) / self.omega