import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import logging
from typing import Literal, Optional, Union, Callable
import random

METRIC_DISTRIBUTIONS = {
    "normal": np.random.normal,
    "uniform": np.random.uniform,
    "exponential": np.random.exponential,
    "binomial": np.random.binomial,
    "poisson": np.random.poisson,
}

class Generic_Generator:
    def __init__(
        self,
        n_actions: int,
        gauss_noise: float = 0.1,
    ) -> None:
        self.n_actions = n_actions
        self.gauss_noise = gauss_noise

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

class Synthetic_AA_test(Generic_Generator):
    def generate_raw_logs(
        self,
        metric_distribution: str,
        n_users: int = 1000,
        min_logs: Optional[int] = None,
        return_as_frame: bool = True
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Генерация данных для AA-теста (без эффекта)"""
        metric_sampler = self._validate_distribution(metric_distribution)
        total_logs = []

        for user_id in range(n_users):
            user_data = self._generate_user_data(user_id, metric_sampler)
            total_logs.append(user_data)

        result = np.vstack(total_logs)

        if return_as_frame:
            return pd.DataFrame(result, columns=["user_id", "metric"])
        return result

class Synthetic_AB_test(Generic_Generator):
    def generate_raw_logs_absolute_effect(
        self,
        metric_distribution: str,
        effect_size: int,
        n_users: int = 1000,
        min_logs: Optional[int] = None,
        return_as_frame: bool = True
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Генерация данных с абсолютным эффектом"""
        metric_sampler = self._validate_distribution(metric_distribution)
        total_logs = []

        for user_id in range(n_users):
            user_data = self._generate_user_data(
                user_id, metric_sampler, effect_size, "absolute"
            )
            total_logs.append(user_data)

        result = np.vstack(total_logs)

        if return_as_frame:
            df = pd.DataFrame(result, columns=["user_id", "metric"])
            return df.sample(frac=1).reset_index(drop=True)
        return np.random.permutation(result)

    def generate_raw_logs_percent_effect(
        self,
        metric_distribution: str,
        effect_size: float,
        n_users: int = 1000,
        min_logs: Optional[int] = None,
        return_as_frame: bool = True
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Генерация данных с относительным эффектом"""
        metric_sampler = self._validate_distribution(metric_distribution)
        total_logs = []

        for user_id in range(n_users):
            user_data = self._generate_user_data(
                user_id, metric_sampler, effect_size, "relative"
            )
            total_logs.append(user_data)

        result = np.vstack(total_logs)

        if return_as_frame:
            df = pd.DataFrame(result, columns=["user_id", "metric"])
            return df.sample(frac=1).reset_index(drop=True)
        return np.random.permutation(result)

    def generate_raw_logs_with_outliers(
        self,
        metric_distribution: str,
        effect_size: float,
        n_users: int = 1000,
        min_logs: Optional[int] = None,
        return_as_frame: bool = True,
        n_outliers: int = 10,
        outlier_type: Literal["positive", "negative", "both"] = "both"
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Генерация данных с выбросами"""
        try:
            clean_df = self.generate_raw_logs_percent_effect(
                metric_distribution=metric_distribution,
                effect_size=effect_size,
                n_users=n_users,
                min_logs=min_logs,
                return_as_frame=True
            )
            if not isinstance(clean_df, pd.DataFrame):
                raise ValueError("Internal error while getting metrics data")
            metric_series = clean_df["metric"]
            if not isinstance(metric_series, pd.DataFrame):
                raise ValueError("Internal error while getting metrics data")
            Q1 = metric_series.quantile(0.25)
            Q3 = metric_series.quantile(0.75)
            IQR = Q3 - Q1


            if outlier_type == "negative":
                outlier_multipliers = np.random.uniform(-1.7, -1.5, n_outliers)
            elif outlier_type == "positive":
                outlier_multipliers = np.random.uniform(1.5, 1.7, n_outliers)
            else:
                outlier_multipliers = np.where(
                    np.random.random(n_outliers) < 0.5,
                    np.random.uniform(-1.7, -1.5, n_outliers),
                    np.random.uniform(1.5, 1.7, n_outliers)
                )


            outlier_values = IQR * outlier_multipliers
            max_user_id = clean_df["user_id"].max()
            outlier_user_ids = np.arange(max_user_id + 1, max_user_id + 1 + n_outliers)


            outliers_df = pd.DataFrame({
                "user_id": outlier_user_ids,
                "metric": outlier_values
            })

            result_df = pd.concat([clean_df, outliers_df], ignore_index=True)
            result_df = result_df.sample(frac=1).reset_index(drop=True)
            
            if return_as_frame:
                return result_df
            return result_df.values
            
        except Exception as e:
            logging.error(f"Error generating data with outliers: {e}")
            raise