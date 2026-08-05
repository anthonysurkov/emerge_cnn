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


class TwoLayerConv(torch.nn.Module):
    def __init__(
        self,
        num_filters_layer_one: int,
        num_filters_layer_two: int,
        kernel_size_layer_one: int,
        kernel_size_layer_two: int
    ):
        super().__init__()
        self.layer_one = torch.nn.Conv1d(
            in_channels=4,
            out_channels=num_filters_layer_one,
            kernel_size=kernel_size_layer_one
        )
        self.layer_two = torch.nn.Conv1d(
            in_channels=num_filters_layer_one,
            out_channels=num_filters_layer_two,
            kernel_size=kernel_size_layer_two
        )
        self.activation = torch.nn.ReLU()

    def forward(self, sequences: torch.Tensor):
        layer_one_out = self.layer_one(sequences)
        active_one_out = self.activation(layer_one_out)
        layer_two_out = self.layer_two(active_one_out)
        active_two_out = self.activation(layer_two_out).flatten(start_dim=1)
        return active_two_out


class DenseHeads(torch.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.config = {"input_size": input_size}
        self.pi_head = torch.nn.Linear(input_size, 1) # classifier head
        self.mu_head = torch.nn.Linear(input_size, 1) # regressor head

    def forward(self, filters: torch.Tensor) -> dict:
        pi = torch.sigmoid(self.pi_head(filters)).squeeze(-1)
        mu = torch.sigmoid(self.mu_head(filters)).squeeze(-1)

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

    def forward(self, filters: torch.Tensor) -> dict:
        hidden = self.hidden(filters)

        pi = torch.sigmoid(self.pi_head(hidden)).squeeze(-1)
        mu = torch.sigmoid(self.mu_head(hidden)).squeeze(-1)

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

    def forward(self, filters: torch.Tensor) -> dict:
        hidden_pi = self.hidden_pi(filters)
        hidden_mu = self.hidden_mu(filters)

        pi = torch.sigmoid(self.pi_head(hidden_pi)).squeeze(-1)
        mu = torch.sigmoid(self.mu_head(hidden_mu)).squeeze(-1)

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
