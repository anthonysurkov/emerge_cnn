import torch


class OneHotFeats(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sequences: torch.Tensor):
        return (
            torch.nn.functional.one_hot(sequences, num_classes=4)
              .float()
              .permute(0, 2, 1)
        )


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


class DenseHeads(torch.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.pi_head = torch.nn.Linear(input_size, 1) # classifier head
        self.mu_head = torch.nn.Linear(input_size, 1)  # regressor head

    def forward(self, filters: torch.Tensor) -> dict:
        pi_out = torch.sigmoid(
            self.pi_head(filters)
        ).squeeze(-1)
        mu_out = torch.sigmoid(
            self.mu_head(filters)
        ).squeeze(-1)
        return {"pi": pi_out, "mu": mu_out}
