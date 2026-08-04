import torch


class OneLayerConv(torch.nn.Module):
    def __init__(self, num_filters: int, kernel_size: int):
        super().__init__()
        self.layer = torch.nn.Conv1d(
            in_channels=4,
            out_channels=num_filters,
            kernel_size=kernel_size
        )
        self.activation = torch.nn.ReLU()


    def forward(self, sequences: torch.Tensor):
        layer_out = self.layer(sequences)
        active_out = self.activation(layer_out).flatten(start_dim=1)
        return active_out
