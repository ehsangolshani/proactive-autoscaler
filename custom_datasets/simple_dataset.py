import torch
from torch.utils.data.dataset import Dataset
import pandas as pd


class SimpleWorkloadDataset(Dataset):
    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path)[['normalized_request_rate']]
        self.data_tensor: torch.Tensor = torch.tensor(data=self.data.values, dtype=torch.float)
        self.data_tensor.contiguous()
        # self.data_tensor = self.data_tensor.view(self.data_tensor.size()[1], self.data_tensor.size()[0])
        self.data_tensor = self.data_tensor.t()
        self.window_size: int = 2

    def __getitem__(self, index: int):
        o = self.data_tensor[:, index:index + self.window_size]
        return o

    def __len__(self):
        return self.data_tensor.size()[1] - self.window_size + 1
