import pytest
import math
import sys
import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from functions import function_factory
from functions import Generic_Function
from functions import FUNCTION_TYPES


class TestFunctionFactory:
    """Тесты для фабрики функций"""

    def test_factory_returns_correct_types(self):
        """Тест, что фабрика возвращает правильные типы функций"""
        linear = function_factory("linear", k=2.0, b=1.0)
        assert isinstance(linear, Generic_Function)
        assert linear.function_type == "linear"
        quadratic = function_factory("quadratic", a=1.0, b=0.0, c=-1.0)
        assert quadratic.function_type == "quadratic"

    def test_factory_unknown_function_type(self):
        """Тест на неизвестный тип функции"""
        with pytest.raises(NotImplementedError):
            function_factory("unknown_type", k=1.0)

    def test_factory_missing_coefficients(self):
        """Тест на отсутствие обязательных коэффициентов"""
        with pytest.raises(ValueError):
            function_factory("linear", k=2.0)
        with pytest.raises(ValueError):
            function_factory("quadratic", a=1.0, b=0.0)

    def test_factory_exponential_lambda_alias(self):
        """Тест обработки lambda -> lambda_ для экспоненциальной функции"""
        exp1 = function_factory("exponential", a=2.0, lambda_=0.5)
        assert exp1._coefficients["lambda_"] == 0.5


class TestLinearFunction:
    """Тесты для линейной функции"""

    def test_linear_creation(self):
        """Тест создания линейной функции"""

        linear = function_factory("linear", k=2.0, b=1.0)
        assert linear.k == 2.0
        assert linear.b == 1.0

    def test_linear_evaluation(self):
        """Тест вычисления линейной функции"""

        linear = function_factory("linear", k=2.0, b=1.0)
        assert linear(0) == 1.0
        assert linear(1) == 3.0
        assert linear(2) == 5.0
        assert linear(-1) == -1.0
    
    def test_linear_call_vs_evaluate(self):
        """Тест, что __call__ и _evaluate дают одинаковый результат"""

        linear = function_factory("linear", k=2.0, b=1.0)
        x = 5
        assert linear(x) == linear._evaluate(x)

    def test_linear_update_coefficients(self):
        """Тест обновления коэффициентов"""

        linear = function_factory("linear", k=2.0, b=1.0)
        linear.set_coefficients(k=3.0, b=0.0)
        assert linear(1) == 3.0


class TestQuadraticFunction:
    """Тесты для квадратичной функции"""

    def test_quadratic_creation(self):
        """Тест создания квадратичной функции"""

        quadratic = function_factory("quadratic", a=1.0, b=-2.0, c=1.0)
        assert quadratic.a == 1.0
        assert quadratic.b == -2.0
        assert quadratic.c == 1.0

    def test_quadratic_evaluation(self):
        """Тест вычисления квадратичной функции"""

        quadratic = function_factory("quadratic", a=1.0, b=0.0, c=0.0)
        assert quadratic(0) == 0.0
        assert quadratic(1) == 1.0
        assert quadratic(2) == 4.0
        assert quadratic(-1) == 1.0

    def test_quadratic_vertex(self):
        """Тест вершины параболы"""

        quadratic = function_factory("quadratic", a=1.0, b=-4.0, c=4.0)
        assert quadratic(2) == 0.0


class TestCubicFunction:
    """Тесты для кубической функции"""

    def test_cubic_creation(self):
        """Тест создания кубической функции"""

        cubic = function_factory("cubic", a=1.0, b=0.0, c=0.0, d=0.0)
        assert cubic.a == 1.0
        assert cubic.d == 0.0
    
    def test_cubic_evaluation(self):
        """Тест вычисления кубической функции"""

        cubic = function_factory("cubic", a=1.0, b=0.0, c=0.0, d=0.0)
        assert cubic(0) == 0.0
        assert cubic(1) == 1.0
        assert cubic(2) == 8.0
        assert cubic(-1) == -1.0


class TestExponentialFunction:
    """Тесты для экспоненциальной функции"""

    def test_exponential_creation(self):
        """Тест создания экспоненциальной функции"""

        exp = function_factory("exponential", a=2.0, lambda_=0.5)
        assert exp.a == 2.0
        assert exp.lambda_ == 0.5
    
    def test_exponential_evaluation(self):
        """Тест вычисления экспоненциальной функции"""

        exp = function_factory("exponential", a=1.0, lambda_=0.0)
        assert exp(0) == 1.0
        assert exp(1) == math.exp(0)  # 1.0
        exp2 = function_factory("exponential", a=2.0, lambda_=1.0)
        assert exp2(0) == 2.0
        assert math.isclose(exp2(1), 2.0 * math.exp(1))


class TestLogarithmFunction:
    """Тесты для логарифмической функции"""

    def test_logarithm_creation(self):
        """Тест создания логарифмической функции"""
        log = function_factory("logarithm", k=1.0, a=10.0, b=0.0)
        assert log.k == 1.0
        assert log.a == 10.0
        assert log.b == 0.0

    def test_logarithm_evaluation(self):
        """Тест вычисления логарифмической функции"""
        log = function_factory("logarithm", k=1.0, a=10.0, b=0.0)
        assert math.isclose(log(100), 2.0)
        log2 = function_factory("logarithm", k=2.0, a=2.0, b=0.0)
        assert math.isclose(log2(8), 6.0)
    
    def test_logarithm_invalid_arguments(self):
        """Тест логарифма с невалидными аргументами"""

        log = function_factory("logarithm", k=1.0, a=10.0, b=0.0)
        with pytest.raises(ValueError):
            log(0)
        with pytest.raises(ValueError):
            log(-1)

    def test_logarithm_validation(self):
        """Тест валидации коэффициентов логарифма"""
        with pytest.raises(ValueError):
            function_factory("logarithm", k=1.0, a=1.0, b=0.0)
        with pytest.raises(ValueError):
            function_factory("logarithm", k=1.0, a=0.0, b=0.0)


class TestTrigonometricFunctions:
    """Тесты для тригонометрических функций"""

    def test_sinus_function(self):
        """Тест синусоидальной функции"""
        sin_func = function_factory("sinus", a=2.0, omega=math.pi)
        assert sin_func(0) == 0.0
        assert math.isclose(sin_func(0.5), 2.0 * math.sin(math.pi * 0.5))

    def test_cosine_function(self):
        """Тест косинусоидальной функции"""
        cos_func = function_factory("cosine", a=2.0, omega=math.pi)
        assert math.isclose(cos_func(0), 2.0)
        assert math.isclose(cos_func(0.5), 2.0 * math.cos(math.pi * 0.5))

    def test_tangent_function(self):
        """Тест тангенсоидальной функции"""
        tan_func = function_factory("tangent", a=1.0, omega=1.0)
        assert math.isclose(tan_func(0), 0.0)
        assert math.isclose(tan_func(math.pi/4), math.tan(math.pi/4))

    def test_cotangent_function(self):
        """Тест котангенсоидальной функции"""
        cotan_func = function_factory("cotangent", a=1.0, omega=1.0)
        assert math.isclose(cotan_func(math.pi/4), 1.0 / math.tan(math.pi/4))

    def test_tangent_undefined_points(self):
        """Тест особых точек тангенса"""
        tan_func = function_factory("tangent", a=1.0, omega=1.0)
        with pytest.raises(ValueError):
            tan_func(math.pi/2)

    def test_cotangent_undefined_points(self):
        """Тест особых точек котангенса"""
        cotan_func = function_factory("cotangent", a=1.0, omega=1.0)
        with pytest.raises(ValueError):
            cotan_func(0)


class TestCoefficientAccess:
    """Тесты доступа к коэффициентам"""

    def test_coefficient_access(self):
        """Тест доступа к коэффициентам как к атрибутам"""
        linear = function_factory("linear", k=2.0, b=1.0)
        assert linear.k == 2.0
        assert linear.b == 1.0
        quadratic = function_factory("quadratic", a=1.0, b=2.0, c=3.0)
        assert quadratic.a == 1.0
        assert quadratic.b == 2.0
        assert quadratic.c == 3.0

    def test_coefficient_update(self):
        """Тест обновления коэффициентов через set_coefficients"""
        linear = function_factory("linear", k=2.0, b=1.0)
        linear.set_coefficients(k=5.0)
        assert linear.k == 5.0
        assert linear.b == 1.0
        linear.set_coefficients(b=10.0)
        assert linear.k == 5.0
        assert linear.b == 10.0
    
    def test_invalid_attribute_access(self):
        """Тест доступа к несуществующим атрибутам"""

        linear = function_factory("linear", k=2.0, b=1.0)
        with pytest.raises(AttributeError):
            _ = linear.nonexistent_attr


class TestFunctionRepresentation:
    """Тесты строкового представления функций"""

    def test_repr(self):
        """Тест строкового представления"""
        linear = function_factory("linear", k=2.0, b=1.0)
        repr_str = repr(linear)
        assert "LinearFunction" in repr_str
        assert "k" in repr_str
        assert "b" in repr_str


def test_all_function_types_covered():
    """Тест, что все типы функций из FUNCTION_TYPES покрыты классами"""

    from functions.function_classes import FUNCTION_CLASSES
    
    for func_type in FUNCTION_TYPES:
        assert func_type in FUNCTION_CLASSES, f"No class implementation for {func_type}"
        assert FUNCTION_CLASSES[func_type] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])