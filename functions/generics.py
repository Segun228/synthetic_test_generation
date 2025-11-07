import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import logging
from typing import Literal, Optional, Union, Callable
import random
from abc import ABC, abstractmethod
from typing import Tuple
import io
import datetime
from typing import Any


class Generic_Function(ABC):
    """Базовый класс для математических функций"""
    
    def __init__(
        self,
        function_type: Literal['linear', 'quadratic', 'cubic', 'exponential', 'logarithm', 'sinus', 'cosine', 'tangent', 'cotangent'],
        **coefficients
    ) -> None:
        super().__setattr__('function_type', function_type)
        super().__setattr__('_coefficients', {})
        
        
        if coefficients:
            self.set_coefficients(**coefficients)

    def set_coefficients(self, **coefficients) -> None:
        """Установка коэффициентов с валидацией"""
        self._validate_coefficients(coefficients)
        self._coefficients.update(coefficients)


    @abstractmethod
    def _validate_coefficients(self, coefficients: dict) -> None:
        """Валидация коэффициентов (должен быть реализован в потомках)"""
        pass

    @abstractmethod
    def _evaluate(self, x: Any) -> Any:
        """Вычисление функции в точке x (должен быть реализован в потомках)"""
        pass

    def __call__(self, x: Any) -> Any:
        """Вызов функции как fun(x)"""
        return self._evaluate(x)

    def __getattr__(self, name: str) -> Any:
        """Доступ к коэффициентам как к атрибутам"""
        if name in self._coefficients:
            return self._coefficients[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Установка атрибутов с защитой коэффициентов"""
        if name in getattr(self, '_coefficients', {}):
            self.set_coefficients(**{name: value})
        else:
            super().__setattr__(name, value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._coefficients})"

    @abstractmethod
    def inverse_function(self, argument) -> Any:
        raise NotImplementedError()

    @property
    def coefficients(self) -> dict:
        """Возвращает копию коэффициентов"""
        return self._coefficients.copy()