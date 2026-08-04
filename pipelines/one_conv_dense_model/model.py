import torch


class OneConvDenseModel(torch.nn.Module):
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
