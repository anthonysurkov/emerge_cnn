import torch
from captum.attr import DeepLift
from modiscolite.tfmodisco import TFMoDISco
from modiscolite.io import save_hdf5

from . import model
from .truth import EmergeDataset, EmergeCNNPaths, load_train_val
from .utils import model_from_checkpoint


class AttrModel(torch.nn.Module):
    def __init__(
        self,
        cnn: model.ConvModelFramework,
        output_name
    ):
        super().__init__()
        self.model = cnn
        self.output_name = output_name
        if isinstance(cnn.conv_block, model.TwoLayerConv):
            self.activation_one = torch.nn.ReLU()
            self.activation_two = torch.nn.ReLU()

    def forward(self, one_hot):
        conv = self.model.conv_block
        if isinstance(conv, model.TwoLayerConv):
            features = self.activation_one(conv.layer_one(one_hot))
            features = self.activation_two(conv.layer_two(features))
            features = features.flatten(start_dim=1)
        else:
            features = conv(one_hot)
        outputs = self.model.heads_block(features)
        return outputs[self.output_name]


def get_attr_scores(
    model: model.ConvModelFramework,
    sequences: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    one_hot = (
        torch.nn.functional.one_hot(
            sequences.to(device),
            num_classes=4
        )
        .permute(0, 2, 1)
        .to(device=device, dtype=dtype)
        .detach()
        .requires_grad_(True)
    )

    baseline = torch.full_like(one_hot, 0.25)

    attr_model = AttrModel(model, output_name="mu")
    deep_lift = DeepLift(attr_model)

    hypothetical_contribs = deep_lift.attribute(
        inputs=one_hot,
        baselines=baseline,
        custom_attribution_func=_hypothetical_contribs
    )

    return one_hot, hypothetical_contribs


def _hypothetical_contribs(
    multipliers: tuple[torch.Tensor, ...],
    inputs: tuple[torch.Tensor, ...],
    baselines: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, ...]:
    del inputs

    hypothetical = []
    for multiplier, baseline in zip(multipliers, baselines):
        baseline_contrib = (multiplier * baseline).sum(
            dim=1,
            keepdim=True
        )
        hypothetical.append(multiplier - baseline_contrib)

    return tuple(hypothetical)


def do_modisco(
    ohe: torch.Tensor,
    hypothetical_contribs: torch.Tensor,
    output_path: str = "temp/modisco_results.h5",
    window_size: int = 6,
    flank_size: int = 2,
    max_seqlets: int = 5000
) -> tuple[list | None, list | None]:
    if ohe.shape != hypothetical_contribs.shape:
        raise ValueError(
            "One-hot sequences and hypothetical contributions must have "
            f"the same shape, got {ohe.shape} and "
            f"{hypothetical_contribs.shape}"
        )
    if ohe.ndim != 3 or ohe.shape[1] != 4:
        raise ValueError(f"Expected tensors shaped (N, 4, L), got {ohe.shape}")

    if flank_size < 1:
        raise ValueError(
            "flank_size must be at least 1 to avoid modisco-lite's "
            "zero-flank empty-slice bug"
        )

    sequence_length = ohe.shape[2]
    if window_size > sequence_length:
        raise ValueError(
            "window_size must not exceed the unpadded sequence length "
            f"({window_size} > {sequence_length})"
        )

    # MoDISco requires a positive flank, but requiring that context from the
    # original sequence would exclude windows at either edge. Add neutral
    # context instead: uniform bases with zero hypothetical contribution.
    # Padding by exactly flank_size keeps every window in the original
    # sequence eligible for seqlet extraction.
    padded_ohe = torch.nn.functional.pad(
        ohe,
        (flank_size, flank_size),
        value=0.25
    )
    padded_hypothetical = torch.nn.functional.pad(
        hypothetical_contribs,
        (flank_size, flank_size),
        value=0.0
    )

    one_hot = padded_ohe.detach().cpu().permute(0, 2, 1).numpy()
    hypothetical = (
        padded_hypothetical
        .detach()
        .cpu()
        .permute(0, 2, 1)
        .numpy()
    )

    pos_patterns, neg_patterns = TFMoDISco(
        one_hot=one_hot,
        hypothetical_contribs=hypothetical,
        sliding_window_size=window_size,
        flank_size=flank_size,
        trim_to_window_size=10,
        initial_flank_to_add=0,
        final_flank_to_add=0,
        max_seqlets_per_metacluster=max_seqlets,
        nearest_neighbors_to_compute=100,
        n_leiden_runs=2,
        verbose=True
    )
    save_hdf5(
        output_path,
        pos_patterns,
        neg_patterns,
        window_size=window_size
    )

    return pos_patterns, neg_patterns


def attr_main(checkpoint_path: str, screen_name="r255x"):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    paths = EmergeCNNPaths(screen_name=screen_name)

    _, val_df = load_train_val(paths)
    val_loader = torch.utils.data.DataLoader(
        EmergeDataset(val_df),
        batch_size=4096,
        shuffle=False
    )

    cnn = model_from_checkpoint(checkpoint)
    cnn.to(device)

    one_hot_batches = []
    hypothetical_batches = []
    for batch in val_loader:
        one_hot, hypothetical = get_attr_scores(
            model=cnn,
            sequences=batch["sequence"]
        )
        one_hot_batches.append(one_hot.detach().cpu())
        hypothetical_batches.append(hypothetical.detach().cpu())

    one_hot = torch.cat(one_hot_batches)
    hypothetical_contribs = torch.cat(hypothetical_batches)
    do_modisco(
        ohe=one_hot,
        hypothetical_contribs=hypothetical_contribs
    )


if __name__ == "__main__":
    attr_main(checkpoint_path="data/current_best/twoconv_sharedhidden_model_k13_k24_f116_f264_h32_ckpt.pt")
