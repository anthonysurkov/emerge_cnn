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
        self.mu_head = torch.nn.Linear(input_size, 1) # regressor head

    def forward(self, filters: torch.Tensor) -> dict:
        pi_out = torch.sigmoid(
            self.pi_head(filters)
        ).squeeze(-1)
        mu_out = torch.sigmoid(
            self.mu_head(filters)
        ).squeeze(-1)
        return {"pi": pi_out, "mu": mu_out}


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
