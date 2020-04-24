import torch
from torch import nn
from workload_predictor.TCN.tcn import TemporalConvNet


class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, sequence_length):
        super(TCNModel, self).__init__()
        self.tcn: TemporalConvNet = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(sequence_length, output_size)
        self.final_relu = nn.ReLU()

    def forward(self, x):
        y1: torch.Tensor = self.tcn(x)
        y1 = y1.squeeze()
        return self.linear(y1)


def load_and_initialize_tcn_workload_model(model_path: str, dropout=0.0, window_size=24) -> TCNModel:
    hidden_units_per_layer = 1  # channel
    levels = 4
    channel_sizes = [hidden_units_per_layer] * levels
    input_channels = 1
    output_size = 1
    kernel_size = 3

    model: TCNModel = TCNModel(input_size=input_channels,
                               output_size=output_size,
                               num_channels=channel_sizes,
                               kernel_size=kernel_size,
                               dropout=dropout,
                               sequence_length=window_size)
    model.load_state_dict(torch.load(model_path))
    return model
