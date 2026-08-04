import torch


class Features(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sequences: torch.Tensor):
        return (
            torch.nn.functional.one_hot(sequences, num_classes=4)
              .float()
              .permute(0, 2, 1)
        )
