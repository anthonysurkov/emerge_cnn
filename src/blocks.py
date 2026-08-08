import torch
from collections.abc import Sequence


class OneHotFeats(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sequences: torch.Tensor):
        return (
            torch.nn.functional.one_hot(sequences, num_classes=4)
              .float()
              .permute(0, 2, 1)
        )


class ConvStack(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        layers: Sequence[tuple[int, int]]
    ):
        super().__init__()
        if not layers:
            raise ValueError("ConvStack needs at least one layer")

        convolutions = []
        current_channels = in_channels
        for out_channels, kernel_size in layers:
            if out_channels < 1:
                raise ValueError("out_channels must be positive")
            if kernel_size < 1:
                raise ValueError("kernel_size must be positive")

            convolutions.append(torch.nn.Conv1d(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size
            ))
            current_channels = out_channels

        self.convolutions = torch.nn.ModuleList(convolutions)
        self.activations = torch.nn.ModuleList(
            torch.nn.ReLU()
            for _ in self.convolutions
        )
        self.out_channels = current_channels

    def output_length(self, input_length: int) -> int:
        length = input_length
        for conv in self.convolutions:
            kernel_size = conv.kernel_size[0]
            stride = conv.stride[0]
            padding = conv.padding[0]
            dilation = conv.dilation[0]

            length = (
                length
                + 2 * padding
                - dilation * (kernel_size - 1)
                - 1
            ) // stride + 1

        if length < 1:
            raise ValueError(
                f"Convolution stack reduces {input_length} below 1"
            )
        return length

    def output_size(self, input_length: int) -> int:
        return self.out_channels * self.output_length(input_length)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs

        for convolution, activation in zip(
            self.convolutions,
            self.activations
        ):
            outputs = activation(convolution(outputs))

        return outputs.flatten(start_dim=1)


class DenseHeads(torch.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.config = {"input_size": input_size}
        self.pi_head = torch.nn.Linear(input_size, 1) # classifier head
        self.mu_head = torch.nn.Linear(input_size, 1) # regressor head
        self.pi_activation = torch.nn.Sigmoid()
        self.mu_activation = torch.nn.Sigmoid()

    def forward(self, filters: torch.Tensor) -> dict:
        pi = self.pi_activation(self.pi_head(filters)).squeeze(-1)
        mu = self.mu_activation(self.mu_head(filters)).squeeze(-1)

        return {"pi": pi, "mu": mu}


class SharedHidden(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.config = {
            "input_size": input_size,
            "hidden_size": hidden_size
        }

        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU()
        )
        self.pi_head = torch.nn.Linear(hidden_size, 1)
        self.mu_head = torch.nn.Linear(hidden_size, 1)
        self.pi_activation = torch.nn.Sigmoid()
        self.mu_activation = torch.nn.Sigmoid()

    def forward(self, filters: torch.Tensor) -> dict:
        hidden = self.hidden(filters)

        pi = self.pi_activation(self.pi_head(hidden)).squeeze(-1)
        mu = self.mu_activation(self.mu_head(hidden)).squeeze(-1)

        return {"pi": pi, "mu": mu}


class TwoSharedHidden(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size_one: int,
        hidden_size_two: int
    ):
        super().__init__()
        self.config = {
            "input_size": input_size,
            "hidden_size_one": hidden_size_one,
            "hidden_size_two": hidden_size_two
        }
        self.hidden_one = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size_one),
            torch.nn.ReLU()
        )
        self.hidden_two = torch.nn.Sequential(
            torch.nn.Linear(hidden_size_one, hidden_size_two),
            torch.nn.ReLU()
        )
        self.pi_head = torch.nn.Linear(hidden_size_two, 1)
        self.mu_head = torch.nn.Linear(hidden_size_two, 1)
        self.pi_activation = torch.nn.Sigmoid()
        self.mu_activation = torch.nn.Sigmoid()

    def forward(self, filters: torch.Tensor) -> dict:
        hidden_one = self.hidden_one(filters)
        hidden_two = self.hidden_two(hidden_one)

        pi = self.pi_activation(self.pi_head(hidden_two)).squeeze(-1)
        mu = self.mu_activation(self.mu_head(hidden_two)).squeeze(-1)

        return {"pi": pi, "mu": mu}


class SplitHidden(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size_pi: int,
        hidden_size_mu: int
    ):
        super().__init__()
        self.config = {
            "input_size": input_size,
            "hidden_size_pi": hidden_size_pi,
            "hidden_size_mu": hidden_size_mu
        }

        self.hidden_pi = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size_pi),
            torch.nn.ReLU()
        )
        self.hidden_mu = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size_mu),
            torch.nn.ReLU()
        )
        self.pi_head = torch.nn.Linear(hidden_size_pi, 1)
        self.mu_head = torch.nn.Linear(hidden_size_mu, 1)
        self.pi_activation = torch.nn.Sigmoid()
        self.mu_activation = torch.nn.Sigmoid()

    def forward(self, filters: torch.Tensor) -> dict:
        hidden_pi = self.hidden_pi(filters)
        hidden_mu = self.hidden_mu(filters)

        pi = self.pi_activation(self.pi_head(hidden_pi)).squeeze(-1)
        mu = self.mu_activation(self.mu_head(hidden_mu)).squeeze(-1)

        return {"pi": pi, "mu": mu}


class ConvModelFramework(torch.nn.Module):
    def __init__(self, encoder_block, conv_block, heads_block, phi_init):
        super().__init__()
        self.encoder_block = encoder_block
        self.conv_block = conv_block
        self.heads_block = heads_block
        self.phi_raw = torch.nn.Parameter(
            torch.tensor(phi_init, dtype=torch.float32)
        )

    def forward(self, sequences: torch.Tensor):
        encoder_out = self.encoder_block(sequences)
        convs_out = self.conv_block(encoder_out)
        out = self.heads_block(convs_out)
        phi = torch.nn.functional.softplus(self.phi_raw)
        out["phi"] = phi
        return out
