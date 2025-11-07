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
from functions import function_factory

class Generic_Time_Series(ABC):
    def __init__(
        self,
        t_start:datetime.datetime,
        t_end:datetime.datetime,
        n_events:int,
        initial_value:float,
    ):
        self.t_start:datetime.datetime = t_start
        self.t_end:datetime.datetime = t_end
        self.n_events:int = n_events
        self.initial_value = initial_value
        self.time_series = self.generate_initial_row()
        self.trend_function = None

    def visualize(
        self,
        metric_name = "Метрика",
        figsize: tuple = (16, 10),
        dpi: int = 300,
        show_plot: bool = True
    )->io.BytesIO|None:
        try:
            time_series = self.time_series
            if time_series is None or time_series.empty:
                raise ValueError("The series is empty, could not generate a proper row")
            plt.figure(figsize=figsize)
            plt.plot(time_series["time"], time_series["metric"], linewidth=1)
            plt.xlabel('Период')
            plt.ylabel(metric_name)
            plt.title('Базовое представление временного ряда')
            plt.legend()
            plt.grid(True, alpha=0.3)
            buffer = io.BytesIO()
            plt.savefig(buffer, dpi=dpi, bbox_inches='tight')
            logging.info("График построен")
            if show_plot:
                plt.show()
                plt.close()
            return buffer
        except Exception as e:
            logging.error(e)
            raise

    def generate_initial_row(self) -> pd.DataFrame:
        if not all([self.t_start, self.t_end, self.n_events]):
            raise ValueError("Missing vital data arguments")
        time_points = [self.t_start + i * (self.t_end - self.t_start) / self.n_events for i in range(self.n_events)]
        self.time_series = pd.DataFrame({
            "time": time_points,
            "metric": [self.initial_value] * self.n_events
        })
        return self.time_series


    def create_noise(
        self,
        noise_level: float = 0.2,  
        noise_type: Literal["uniform", "gaussian"] = "uniform",
        return_time_series = False,
    ) -> pd.DataFrame:
        time_series = self.time_series
        if time_series is None or time_series.empty:
            raise ValueError("Error while creating noise, initial time series is empty")
        
        multiplier = max(time_series["metric"].std(), time_series["metric"].mean(), 100)
        n_points = len(time_series)
        
        if noise_type == "uniform":
            noise_vector = np.random.uniform(
                low=-noise_level * multiplier,
                high=noise_level * multiplier,
                size=n_points
            )
        elif noise_type == "gaussian":
            noise_vector = np.random.normal(
                loc=0,
                scale=noise_level * multiplier,
                size=n_points
            )
        else:
            raise ValueError("noise_type must be 'uniform' or 'gaussian'")
        time_series = time_series.copy()
        time_series["metric"] += noise_vector
        self.time_series = time_series
        return self.time_series

    def create_trend(
        self,
        function_type: Literal['linear', 'quadratic', 'cubic', 'exponential', 'logarithm', 'sinus', 'cosine', 'tangent', 'cotangent'],
        return_time_series=False,
        trend_strength: float = 0.3,
        **coefficients
    ):
        try:
            new_function = function_factory(function_type=function_type, **coefficients)
            self.trend_function = new_function
            n_points = len(self.time_series)
            if function_type in ['sinus', 'cosine', 'tangent', 'cotangent']:
                x_values = np.linspace(0, 4 * np.pi, n_points)
            elif function_type == 'exponential':
                x_values = np.linspace(0, 2, n_points)
            elif function_type == 'logarithm':
                x_values = np.linspace(1, 10, n_points)
            else:
                x_values = np.linspace(-2, 2, n_points)
            trend_values = pd.Series([new_function(x) for x in x_values])
            current_mean = self.time_series["metric"].mean()
            current_std = self.time_series["metric"].std() 
            if current_mean == 0:
                current_mean+=1
            if current_std == 0:
                current_std+=1
            trend_normalized = (trend_values - trend_values.mean()) / trend_values.std()
            trend_scaled = trend_normalized * current_std * trend_strength
            self.time_series["metric"] += trend_scaled
            if return_time_series:
                return self.time_series
        except Exception as e:
            logging.exception(e)
            raise

    def create_season(self):
        raise NotImplementedError()

    def create_cycle(self):
        raise NotImplementedError()

    def create_outlier(self):
        raise NotImplementedError()