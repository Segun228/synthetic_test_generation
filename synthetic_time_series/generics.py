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
        noise_type: Literal["uniform", "gaussian"] = "uniform"
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
        return time_series

    def create_trend(
        self
    ):
        
        raise NotImplementedError()

    def create_season(self):
        raise NotImplementedError()

    def create_cycle(self):
        raise NotImplementedError()

    def create_outlier(self):
        raise NotImplementedError()