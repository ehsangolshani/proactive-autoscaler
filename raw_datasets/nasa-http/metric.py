import pandas as pd


def augment_naive_cpu_intensive_metrics(df: pd.DataFrame, interval: int = 60):
    request_rate = df['request_rate']
    previous_request_rate = df['request_rate'].shift(periods=1)
    previous_cpu_utilization = df['cpu_utilization'].shift(periods=1)
    previous_memory_utilization = df['memory_utilization'].shift(periods=1)
    previous_gpu_utilization = df['gpu_utilization'].shift(periods=1)

    previous_request_rate.fillna(0, inplace=True)
    previous_cpu_utilization.fillna(0, inplace=True)
    previous_memory_utilization.fillna(0, inplace=True)
    previous_gpu_utilization.fillna(0, inplace=True)

    x = (request_rate / interval) * 5

    df['cpu_utilization'] = (4 * x ** 3) - (3 * x ** 2) + (3 * x) + 250
    df['memory_utilization'] = 30 + x * 50
    df['gpu_utilization'] = 5 + 0.1 * x


def augment_cpu_intensive_metrics(df: pd.DataFrame, interval: int = 60):
    request_rate = df['request_rate']
    previous_request_rate = df['request_rate'].shift(periods=1)
    previous_cpu_utilization = df['cpu_utilization'].shift(periods=1)
    previous_memory_utilization = df['memory_utilization'].shift(periods=1)
    previous_gpu_utilization = df['gpu_utilization'].shift(periods=1)

    previous_request_rate.fillna(0, inplace=True)
    previous_cpu_utilization.fillna(0, inplace=True)
    previous_memory_utilization.fillna(0, inplace=True)
    previous_gpu_utilization.fillna(0, inplace=True)

    x = request_rate / 8

    df['cpu_utilization'] = (2 * (
            x / 2 + previous_cpu_utilization / 900 + previous_memory_utilization / 400) ** 3) - (1 * (
            x / 3 + previous_cpu_utilization / 700 + previous_memory_utilization / 300) ** 2) + (
                                    3 * (x / 2 + previous_cpu_utilization / 700)) + 50
    df['memory_utilization'] = (x / 5) ** 4 + (previous_memory_utilization / 100 + x * 5) + 100
    df['gpu_utilization'] = 50 + ((0.6 * x) ** 2 + previous_gpu_utilization) / 2


def augment_memory_intensive_metrics(df: pd.DataFrame, interval: int = 60):
    raise NotImplementedError


def augment_gpu_intensive_metrics(df: pd.DataFrame, interval: int = 60):
    raise NotImplementedError


def augment_io_intensive_metrics(df: pd.DataFrame, interval: int = 60):
    raise NotImplementedError
