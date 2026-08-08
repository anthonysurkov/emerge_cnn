import torch
import numpy as np
from pathlib import Path
from captum.attr import DeepLift
from modiscolite.tfmodisco import TFMoDISco
from modiscolite.io import save_hdf5

from . import model
from .truth import EmergeDataset, EmergeCNNPaths, load_train_val
from . import utils


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

    hyp_contribs = deep_lift.attribute(
        inputs=one_hot,
        baselines=baseline,
        custom_attribution_func=_hyp_contribs
    )

    return one_hot, hyp_contribs


def _hyp_contribs(
    multipliers: tuple[torch.Tensor, ...],
    inputs: tuple[torch.Tensor, ...],
    baselines: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, ...]:
    del inputs

    hyp = []
    for multiplier, baseline in zip(multipliers, baselines):
        baseline_contrib = (multiplier * baseline).sum(
            dim=1,
            keepdim=True
        )
        hyp.append(multiplier - baseline_contrib)

    return tuple(hyp)

def calc_padded_ohe(ohe: torch.Tensor, size: int) -> np.ndarray:
    return (
        torch.nn.functional.pad(ohe, (size, size), value=0.25)
          .detach()
          .cpu()
          .permute(0, 2, 1)
          .numpy()
    )

def calc_padded_hyp(hyps: torch.Tensor, size: int) -> np.ndarray:
    return (
        torch.nn.functional.pad(hyps, (size, size), value=0.0)
          .detach()
          .cpu()
          .permute(0, 2, 1)
          .numpy()
    )

def do_modisco(
    ohe: torch.Tensor,
    hyp_contribs: torch.Tensor,
    outpath: str | None = None,
    window_size: int = 6,
    flank_size: int = 2,
    max_seqlets: int = 5000
) -> tuple[list | None, list | None]:
    if ohe.shape != hyp_contribs.shape:
        raise ValueError(
            "One-hot sequences and hypothetical contributions must have "
            f"the same shape, got {ohe.shape} and "
            f"{hyp_contribs.shape}"
        )
    if ohe.ndim != 3 or ohe.shape[1] != 4:
        raise ValueError(f"Expected tensors shaped (N, 4, L), got {ohe.shape}")
    if flank_size < 1:
        raise ValueError(
            "flank_size must be at least 1 to avoid modisco-lite's "
            "zero-flank empty-slice bug"
        )
    if (2 * flank_size) + window_size != 10:
        raise ValueError(
            "flank_size + window_size must be 10 to match modisco-lite's "
            "output length for EMERGe"
        )
    sequence_length = ohe.shape[2]
    if window_size > sequence_length:
        raise ValueError(
            "window_size must not exceed the unpadded sequence length "
            f"({window_size} > {sequence_length})"
        )

    ohe_np: np.ndarray = calc_padded_ohe(ohe=ohe, size=flank_size)
    hyp_np: np.ndarray = calc_padded_hyp(hyps=hyp_contribs, size=flank_size)

    pos_patterns, neg_patterns = TFMoDISco(
        one_hot=ohe_np,
        hypothetical_contribs=hyp_np,
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
    if outpath is not None:
        save_hdf5(outpath, pos_patterns, neg_patterns, window_size=10)

    return pos_patterns, neg_patterns

def calc_hyp_batches(
    val_loader: torch.utils.data.DataLoader,
    cnn: torch.nn.Module
) -> tuple[list, list]:
    ohe_batches = []
    hyp_batches = []
    for batch in val_loader:
        ohe, hyp = get_attr_scores(
            model=cnn,
            sequences=batch["sequence"]
        )
        ohe_batches.append(ohe.detach().cpu())
        hyp_batches.append(hyp.detach().cpu())
    return ohe_batches, hyp_batches

def attr_main(checkpoint_path: str, screen_name="r255x"):
    device = utils.get_device()

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

    cnn = utils.model_from_checkpoint(checkpoint)
    cnn.to(device)

    ohe_batches, hyp_batches = calc_hyp_batches(val_loader, cnn)
    ohe = torch.cat(ohe_batches)
    hyp_contribs = torch.cat(hyp_batches)

    pos_patterns, neg_patterns = do_modisco(ohe=ohe, hyp_contribs=hyp_contribs)
    print(pos_patterns)


if __name__ == "__main__":
    attr_main(checkpoint_path="data/current_best/twoconv_sharedhidden_model_k13_k24_f116_f264_h32_ckpt.pt")
