import torch
from torch.nn import Linear


class DenseHeads(torch.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.pi_head = Linear(input_size, 1) # classifier head
        self.mu_head = Linear(input_size, 1)  # regressor head

    def forward(self, filters: torch.Tensor) -> dict:
        pi_out = torch.sigmoid(
            self.pi_head(filters)
        ).squeeze(-1)
        mu_out = torch.sigmoid(
            self.mu_head(filters)
        ).squeeze(-1)
        return {"pi": pi_out, "mu": mu_out}
