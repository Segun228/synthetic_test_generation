import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


METRIC_DISTRIBUTIONS = {
    "normal":np.random.normal,
    "uniform":np.random.uniform,
    "exponential":np.random.exponential,
    "binomial":np.random.binomial,
    "poisson":np.random.poisson,
}


class Synthetic_AA_test:
    def __init__(
        self,
        metric_distribution:str,
        n_actions:int,
        gauss_noise:float = 0.1,
    ) -> None:
        self.n_actions = n_actions,
        self.gauss_noise = gauss_noise

    def generate_raw_logs(
        self,
        metric_distribution,
        n_users = 1000,
        min_logs = None,
        return_as_frame = True
    )->np.ndarray|pd.DataFrame|None:
        total_logs = []
        if not metric_distribution or metric_distribution not in METRIC_DISTRIBUTIONS.keys():
            raise KeyError("Currently we do not support this type of distribution")
        metric_sampler = METRIC_DISTRIBUTIONS.get(metric_distribution)
        if not metric_sampler:
            raise ValueError("Corresponding sampler was not foud")
        for i in range(n_users):
            n_acts = np.random.poisson(lam=self.n_actions)
            user_results = metric_sampler(n_acts).reshape(n_acts, -1) 
            noise_vector = np.random.uniform(
                low = 1 - self.gauss_noise,
                high = 1 + self.gauss_noise,
                size = n_acts
            )
            user_results *= noise_vector
            user_id_vector = np.array([i for _ in range(n_acts)]).reshape(n_acts, 2)
            user_results = np.concat(
                (user_id_vector, user_results),
                axis=1
            )
            total_logs.append(user_results)
        if return_as_frame:
            df = pd.DataFrame(
                total_logs,
                index = None
            )
            df.columns = ["user_id", "metric"]
            return df
        return np.concat(total_logs, axis=0)


class Synthetic_AB_test:
    def __init__(
        self,
        metric_distribution:str,
        n_actions:int,
        gauss_noise:float = 0.1,
    ) -> None:
        self.n_actions = n_actions,
        self.gauss_noise = gauss_noise

    def generate_raw_logs_absolute_effect(
        self,
        metric_distribution,
        effect_size:int,
        n_users = 1000,
        min_logs = None,
        return_as_frame = True
    )->np.ndarray|pd.DataFrame|None:
        total_logs = []
        if not metric_distribution or metric_distribution not in METRIC_DISTRIBUTIONS.keys():
            raise KeyError("Currently we do not support this type of distribution")
        metric_sampler = METRIC_DISTRIBUTIONS.get(metric_distribution)
        if not metric_sampler:
            raise ValueError("Corresponding sampler was not foud")
        for i in range(n_users):
            n_acts = np.random.poisson(lam=self.n_actions)
            user_results = metric_sampler(n_acts).reshape(n_acts, -1) + effect_size
            noise_vector = np.random.uniform(
                low = 1 - self.gauss_noise,
                high = 1 + self.gauss_noise,
                size = n_acts
            )
            user_results *= noise_vector
            user_id_vector = np.array([i for _ in range(n_acts)]).reshape(n_acts, 2)
            user_results = np.concat(
                (user_id_vector, user_results),
                axis=1
            )
            total_logs.append(user_results)
        if return_as_frame:
            df = pd.DataFrame(
                total_logs,
                index = None
            )
            df.columns = ["user_id", "metric"]
            return df
        return np.concat(total_logs, axis=0)


    def generate_raw_logs_uplift_effect(
        self,
        metric_distribution,
        effect_size:float,
        n_users = 1000,
        min_logs = None,
        return_as_frame = True
    )->np.ndarray|pd.DataFrame|None:
        total_logs = []
        if not metric_distribution or metric_distribution not in METRIC_DISTRIBUTIONS.keys():
            raise KeyError("Currently we do not support this type of distribution")
        metric_sampler = METRIC_DISTRIBUTIONS.get(metric_distribution)
        if not metric_sampler:
            raise ValueError("Corresponding sampler was not foud")
        for i in range(n_users):
            n_acts = np.random.poisson(lam=self.n_actions)
            user_results = metric_sampler(n_acts).reshape(n_acts, -1) * (1 + effect_size)
            noise_vector = np.random.uniform(
                low = 1 - self.gauss_noise,
                high = 1 + self.gauss_noise,
                size = n_acts
            )
            user_results *= noise_vector
            user_id_vector = np.array([i for _ in range(n_acts)]).reshape(n_acts, 2)
            user_results = np.concat(
                (user_id_vector, user_results),
                axis=1
            )
            total_logs.append(user_results)
        if return_as_frame:
            df = pd.DataFrame(
                total_logs,
                index = None
            )
            df.columns = ["user_id", "metric"]
            return df
        return np.concat(total_logs, axis=0)