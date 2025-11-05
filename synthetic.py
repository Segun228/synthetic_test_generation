import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import logging
from typing import Literal, Optional, Union, Callable
import random
from .generics import Generic_Generator
import io



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

    def generate_test_control_groups(
        self,
        test_size,
        control_size,
        metric_distribution
    ):
        try:
            test_group = self.generate_raw_logs(
                metric_distribution=metric_distribution,
                n_users=test_size,
                return_as_frame=True
            )
            control_group = self.generate_raw_logs(
                metric_distribution=metric_distribution,
                n_users=control_size,
                return_as_frame=True
            )
            self.test_group = test_group
            self.control_group = control_group
            return test_group, control_group
        except Exception as e:
            logging.error(e)
            raise

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

    def generate_test_control_groups(
        self,
        test_size,
        control_size,
        metric_distribution,
        effect_size,
        effect_type:Literal["absolute", "percent"]
    ):
        try:
            if effect_type == "absolute":
                test_group = self.generate_raw_logs_absolute_effect(
                    metric_distribution=metric_distribution,
                    n_users=test_size,
                    return_as_frame=True,
                    effect_size=effect_size
                )
                control_group = self.generate_raw_logs_absolute_effect(
                    metric_distribution=metric_distribution,
                    n_users=control_size,
                    return_as_frame=True,
                    effect_size=effect_size
                )
            else:
                test_group = self.generate_raw_logs_percent_effect(
                    metric_distribution=metric_distribution,
                    n_users=test_size,
                    return_as_frame=True,
                    effect_size=effect_size
                )
                control_group = self.generate_raw_logs_percent_effect(
                    metric_distribution=metric_distribution,
                    n_users=control_size,
                    return_as_frame=True,
                    effect_size=effect_size
                )
            self.test_group = test_group
            self.control_group = control_group
            return test_group, control_group
        except Exception as e:
            logging.error(e)
            raise

    def add_outliers(
        self,
        test_group: pd.DataFrame,
        control_group: pd.DataFrame,
        return_as_frame: bool = True,
        n_outliers: int = 10,
        outlier_type: Literal["positive", "negative", "both"] = "both"
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Добавление выбросов в объединенные данные test и control групп"""
        test_group["group"] = "test"
        control_group["group"] = "control"
        df_joint = pd.concat([test_group, control_group], axis=0)
        try:
            metric_series = df_joint["metric"]
            Q1 = metric_series.quantile(0.25)
            Q3 = metric_series.quantile(0.75)
            IQR = Q3 - Q1
            if outlier_type == "negative":
                outlier_multipliers = np.random.uniform(-1.7, -1.5, n_outliers)
            elif outlier_type == "positive":
                outlier_multipliers = np.random.uniform(1.5, 1.7, n_outliers)
            else:  # "both"
                outlier_multipliers = np.where(
                    np.random.random(n_outliers) < 0.5,
                    np.random.uniform(-1.7, -1.5, n_outliers),
                    np.random.uniform(1.5, 1.7, n_outliers)
                )
            outlier_values = Q3 + IQR * outlier_multipliers
            max_user_id = df_joint["user_id"].max()
            outlier_user_ids = np.arange(max_user_id + 1, max_user_id + 1 + n_outliers)
            test_size = len(test_group)
            control_size = len(control_group)
            total_size = test_size + control_size
            n_test_outliers = int(n_outliers * (test_size / total_size))
            n_control_outliers = n_outliers - n_test_outliers
            outliers_data = []
            if n_test_outliers > 0:
                test_outlier_ids = outlier_user_ids[:n_test_outliers]
                test_outlier_values = outlier_values[:n_test_outliers]
                outliers_data.extend([
                    {"user_id": uid, "metric": val, "group": "test"} 
                    for uid, val in zip(test_outlier_ids, test_outlier_values)
                ])
            if n_control_outliers > 0:
                control_outlier_ids = outlier_user_ids[n_test_outliers:n_test_outliers + n_control_outliers]
                control_outlier_values = outlier_values[n_test_outliers:n_test_outliers + n_control_outliers]
                outliers_data.extend([
                    {"user_id": uid, "metric": val, "group": "control"} 
                    for uid, val in zip(control_outlier_ids, control_outlier_values)
                ])
            outliers_df = pd.DataFrame(outliers_data)
            df_with_outliers = pd.concat([df_joint, outliers_df], ignore_index=True)
            df_with_outliers = df_with_outliers.sample(frac=1).reset_index(drop=True)
            if return_as_frame:
                return df_with_outliers
            return df_with_outliers.values
        except Exception as e:
            logging.error(f"Error adding outliers to joint data: {e}")
            raise

    def generate_test_control_groups_with_outliers(
        self,
        test_size,
        control_size,
        metric_distribution,
        effect_size,
        effect_type:Literal["absolute", "percent"],
        return_as_frame: bool = True,
        n_outliers: int = 10,
        outlier_type: Literal["positive", "negative", "both"] = "both"
    ):
        try:
            pass#TODO
        except Exception as e:
            logging.error(e)
            raise

class Multi_metric_generator(Generic_Generator):
    pass


class Synthetic_MAB_test(Generic_Generator):
    pass




class Synthetic_conversion_metrics(Generic_Generator):
    pass


class Synthetic_ratio_metrics(Generic_Generator):
    pass


