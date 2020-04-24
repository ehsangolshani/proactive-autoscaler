from typing import List

import torch
from torch.utils.data.dataset import Dataset
import pandas as pd


class WindowedDataset(Dataset):
    def __init__(self, csv_path: str, window_size: int):
        tmp_df: pd.DataFrame = pd.read_csv(csv_path)[
            ['request_rate',
             'cpu_utilization',
             'memory_utilization',
             'gpu_utilization']]

        self.min_request_rate = tmp_df['request_rate'].min()
        self.max_request_rate = tmp_df['request_rate'].max()
        self.min_cpu_utilization = tmp_df['cpu_utilization'].min()
        self.max_cpu_utilization = tmp_df['cpu_utilization'].max()
        self.min_memory_utilization = tmp_df['memory_utilization'].min()
        self.max_memory_utilization = tmp_df['memory_utilization'].max()
        self.min_gpu_utilization = tmp_df['gpu_utilization'].min()
        self.max_gpu_utilization = tmp_df['gpu_utilization'].max()

        del tmp_df

        self.data: pd.DataFrame = pd.read_csv(csv_path)[
            ['normalized_request_rate',
             'normalized_cpu_utilization',
             'normalized_memory_utilization',
             'normalized_gpu_utilization']]
        self.data_tensor: torch.Tensor = torch.tensor(data=self.data.values, dtype=torch.float)
        self.data_tensor = self.data_tensor.contiguous()
        self.data_tensor = self.data_tensor.t()
        self.window_size: int = window_size

    def __getitem__(self, index: int):
        o = self.data_tensor[:, index:index + self.window_size]
        return o

    def __len__(self):
        return self.data_tensor.size()[1] - self.window_size + 1
