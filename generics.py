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

METRIC_DISTRIBUTIONS = {
    "normal": np.random.normal,
    "uniform": np.random.uniform,
    "exponential": np.random.exponential,
    "binomial": np.random.binomial,
    "poisson": np.random.poisson,
}


class Generic_Generator(ABC):
    def __init__(
        self,
        n_actions: int,
        gauss_noise: float = 0.1,
    ) -> None:
        self.n_actions = n_actions
        self.gauss_noise = gauss_noise
        self.test_group = None
        self.control_group = None

    def _validate_distribution(self, metric_distribution: str) -> Callable:
        """Валидация распределения и получение сэмплера"""
        if metric_distribution not in METRIC_DISTRIBUTIONS:
            raise KeyError(f"Unsupported distribution: {metric_distribution}")
        return METRIC_DISTRIBUTIONS[metric_distribution]

    def _generate_user_data(
            self, 
            user_id: int, 
            metric_sampler: Callable, 
            effect_size: Optional[Union[int, float]] = None, 
            effect_type: str = "absolute") -> np.ndarray:
        """Генерация данных для одного пользователя"""
        n_acts = np.random.poisson(lam=self.n_actions)

        user_results = metric_sampler(size=n_acts).reshape(-1, 1)

        if effect_size is not None:
            if effect_type == "absolute":
                user_results += effect_size
            elif effect_type == "relative":
                user_results *= (1 + effect_size)

        noise_vector = np.random.uniform(
            low=1 - self.gauss_noise,
            high=1 + self.gauss_noise,
            size=n_acts
        ).reshape(-1, 1)
        user_results *= noise_vector
        user_id_vector = np.full((n_acts, 1), user_id)
        user_data = np.concatenate((user_id_vector, user_results), axis=1)
        return user_data

    @abstractmethod
    def generate_test_control_groups(
        self,
        test_size,
        control_size,
    )->Tuple[pd.DataFrame, pd.DataFrame]:
        pass

    def visualize_groups(self) -> io.BytesIO | None:
        try:
            plt.figure(figsize=(16, 10))
            if self.test_group is None or self.control_group is None: 
                logging.info("Nothing to visualize")
                return
            if isinstance(self.test_group, pd.DataFrame) and isinstance(self.control_group, pd.DataFrame):
                plt.hist(self.test_group.loc["metric"], bins="auto", alpha=0.6, label='Test group', color='blue', edgecolor='black')
                plt.hist(self.control_group.loc["metric"], bins="auto", alpha=0.6, label='Control group', color='red', edgecolor='black')
            else:
                plt.hist(self.test_group, bins="auto", alpha=0.6, label='Test group', color='blue', edgecolor='black')
                plt.hist(self.control_group, bins="auto", alpha=0.6, label='Control group', color='red', edgecolor='black')
            plt.xlabel('Значения')
            plt.ylabel('Частота')
            plt.title('Сравнение распределений: Тест vs Контроль')
            plt.legend()
            plt.grid(True, alpha=0.3)
            buffer = io.BytesIO()
            plt.savefig(buffer, dpi=300, bbox_inches='tight')
            logging.info("График построен")
            plt.show()
            plt.close()
            return buffer
        except Exception as e:
            logging.error(e)
            raise




class Generic_Time_Series(ABC):
    def __init__(
        self,
        t_start,
        t_end,
        n_
    ):
        pass