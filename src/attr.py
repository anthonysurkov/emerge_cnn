import argparse

import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from captum.attr import DeepLift
from modiscolite.io import save_hdf5
from modiscolite.core import Seqlet, SeqletSet

from . import model
from .modisco import tfmodisco_forward_only
from .truth import (
    EmergeDataset,
    EmergeCNNPaths,
    load_stratified_kfolds,
    load_train_val,
)
from .training import fold_checkpoint_paths
from . import utils


ATTR_BATCH_SIZE = 4096
DEFAULT_BASE_CHECKPOINT_PATH = Path(
    "data/"
    "eval_conv2-f1-32-k1-4-f2-128-k2-4-onehead-32-alpha-0.25.pt"
)


@dataclass
class ModiscoCluster():
    identity: str
    ppm: np.ndarray
    cwm: np.ndarray
    df: pd.DataFrame


class AttrModel(torch.nn.Module):
    def __init__(
        self,
        cnn: model.ConvModelFramework,
        output_name
    ):
        super().__init__()
        self.model = cnn
        self.output_name = output_name

    def forward(self, one_hot):
        features = self.model.conv_block(one_hot)
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
    outpath: str | Path | None = None,
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

    # EMERGe positions are stranded; reverse complements are distinct inputs.
    pos_patterns, neg_patterns = tfmodisco_forward_only(
        one_hot=ohe_np,
        hypothetical_contribs=hyp_np,
        sliding_window_size=window_size,
        flank_size=flank_size,
        trim_to_window_size=10,
        initial_flank_to_add=0,
        final_flank_to_add=0,
        max_seqlets_per_metacluster=max_seqlets,
        min_overlap_while_sliding=1.0,
        nearest_neighbors_to_compute=100,
        n_leiden_runs=2,
        verbose=True
    )
    if outpath is not None:
        save_hdf5(outpath, pos_patterns, neg_patterns, window_size=10)

    return pos_patterns, neg_patterns

def calc_hyp_batches(
    data_loader: torch.utils.data.DataLoader,
    cnn: torch.nn.Module
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    ohe_batches: list[torch.Tensor] = []
    hyp_batches: list[torch.Tensor] = []
    for batch in data_loader:
        ohe, hyp = get_attr_scores(
            model=cnn,
            sequences=batch["sequence"]
        )
        ohe_batches.append(ohe.detach().cpu())
        hyp_batches.append(hyp.detach().cpu())
    return ohe_batches, hyp_batches

def decode_seqlet(seqlet: Seqlet) -> str:
    bases = np.array(["A","C","G","T"])
    return "".join(
        "N" if np.allclose(row, 0.25) else bases[np.argmax(row)]
        for row in seqlet.sequence
    )

def get_seqlets_emerge(
    seqlets: list[Seqlet],
    df: pd.DataFrame
) -> pd.DataFrame:
    example_indices = np.fromiter(
        (seqlet.example_idx for seqlet in seqlets),
        dtype=np.int64,
        count=len(seqlets)
    )
    if ((example_indices < 0) | (example_indices >= len(df))).any():
        raise IndexError("Seqlet example index is outside the source dataframe")

    # A seqlet's example_idx refers directly to the attributed input row. Using
    # it avoids an RNA/DNA alphabet mismatch (U in EMERGe data versus T in
    # TF-MoDISco's decoded sequence) and also handles reverse-complemented
    # seqlets without changing the source guide identity.
    dflet = df.iloc[example_indices].copy()
    dflet["contrib"] = [
        float(np.sum(seqlet.contrib_scores))
        for seqlet in seqlets
    ]
    return (
        dflet.sort_values("contrib", ascending=False)
             .reset_index(drop=True)
    )

def process_modisco_patterns(
    patterns: list[SeqletSet],
    df: pd.DataFrame,
    id_prefix: str
) -> list[ModiscoCluster]:
    return [
        ModiscoCluster(
            identity=f"{id_prefix}{i}",
            ppm=pattern.sequence.copy(),
            cwm=pattern.contrib_scores.copy(),
            df=get_seqlets_emerge(pattern.seqlets, df)
        )
        for i, pattern in enumerate(patterns)
    ]

def attr_one(
    checkpoint_path: str | Path,
    outpath: str | Path | None = None,
    screen_name: str = "r255x"
) -> tuple[list[ModiscoCluster], list[ModiscoCluster]]:
    paths = EmergeCNNPaths(screen_name=screen_name)
    _, val_df = load_train_val(paths)
    val_loader = torch.utils.data.DataLoader(
        EmergeDataset(val_df),
        batch_size=ATTR_BATCH_SIZE,
        shuffle=False
    )

    cnn, _ = utils.load_model(checkpoint_path)

    ohe_batches, hyp_batches = calc_hyp_batches(val_loader, cnn)
    if not ohe_batches:
        raise ValueError("Cannot compute attributions for an empty dataset")
    ohe = torch.cat(ohe_batches)
    hyp_contribs = torch.cat(hyp_batches)

    pos_patterns, neg_patterns = do_modisco(
        ohe=ohe,
        hyp_contribs=hyp_contribs,
        outpath=outpath
    )
    pos_clusts: list[ModiscoCluster] = process_modisco_patterns(
        patterns=pos_patterns,
        df=val_df,
        id_prefix="c"
    )
    neg_clusts: list[ModiscoCluster] = process_modisco_patterns(
        patterns=neg_patterns,
        df=val_df,
        id_prefix="n"
    )
    return pos_clusts, neg_clusts

def _get_reference_mu(
    cnn: torch.nn.Module,
    sequence_length: int = 10
) -> float:
    cnn.eval()

    parameter = next(cnn.parameters())
    baseline = torch.full(
        (1, 4, sequence_length),
        0.25,
        device=parameter.device,
        dtype=parameter.dtype
    )
    attr_model = AttrModel(cnn, output_name="mu")

    with torch.inference_mode():
        reference_mu = attr_model(baseline).item()

    if not np.isfinite(reference_mu) or reference_mu <= 1e-8:
        raise ValueError(
            f"Ref mu too small for normalization: {reference_mu}"
        )

    return reference_mu

def attr_kfolds(
    base_checkpoint_path: str | Path,
    base_out_path: str | Path | None = None,
    screen_name: str = "r255x",
    *,
    n_splits: int = 10,
    sequence_length: int = 10
) -> tuple[list[ModiscoCluster], list[ModiscoCluster]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")

    paths = EmergeCNNPaths(screen_name=screen_name)
    screen_df = pd.read_csv(
        paths.screen_path,
        usecols=["5to3", "n", "k", "mle"]
    )
    sequence_lengths = screen_df["5to3"].str.len()
    if not sequence_lengths.eq(sequence_length).all():
        raise ValueError(
            "All attribution sequences must have length "
            f"{sequence_length}"
        )
    if screen_df.empty:
        raise ValueError("Cannot compute attributions for an empty dataset")

    folds = load_stratified_kfolds(
        paths=paths,
        n_splits=n_splits
    )
    ohe: torch.Tensor | None = None
    normalized_hyp: torch.Tensor | None = None
    attributed = torch.zeros(len(screen_df), dtype=torch.bool)

    checkpoint_paths = fold_checkpoint_paths(
        base_checkpoint_path,
        n_splits=n_splits
    )
    for (_, val_df), checkpoint_path in zip(
        folds,
        checkpoint_paths,
        strict=True
    ):
        cnn, checkpoint = utils.load_model(checkpoint_path)
        reference_mu = _get_reference_mu(
            cnn,
            sequence_length=sequence_length
        )
        val_loader = torch.utils.data.DataLoader(
            EmergeDataset(val_df),
            batch_size=ATTR_BATCH_SIZE,
            shuffle=False
        )
        val_indices = torch.tensor(
            val_df.index.to_numpy(),
            dtype=torch.long
        )
        offset = 0
        for batch in val_loader:
            fold_ohe, fold_hyp = get_attr_scores(
                model=cnn,
                sequences=batch["sequence"]
            )
            fold_ohe = fold_ohe.detach().cpu()
            fold_hyp = fold_hyp.detach().cpu()
            batch_end = offset + len(fold_ohe)

            if ohe is None:
                attribution_shape = (
                    len(screen_df),
                    fold_ohe.shape[1],
                    fold_ohe.shape[2]
                )
                ohe = torch.empty(
                    attribution_shape,
                    dtype=fold_ohe.dtype
                )
                normalized_hyp = torch.zeros(
                    attribution_shape,
                    dtype=fold_hyp.dtype
                )

            batch_indices = val_indices[offset:batch_end]
            if attributed[batch_indices].any():
                raise RuntimeError(
                    "A sequence appears in more than one validation fold"
                )

            ohe[batch_indices] = fold_ohe
            normalized_hyp[batch_indices] = fold_hyp / reference_mu
            attributed[batch_indices] = True
            offset = batch_end

        if offset != len(val_df):
            raise RuntimeError(
                f"Attributed {offset} of {len(val_df)} fold sequences"
            )

        del cnn, checkpoint

    if ohe is None or normalized_hyp is None:
        raise RuntimeError("No fold attributions were computed")
    if not attributed.all():
        missing = int((~attributed).sum().item())
        raise RuntimeError(
            f"Validation folds did not cover {missing} sequences"
        )

    pos_patterns, neg_patterns = do_modisco(
        ohe=ohe,
        hyp_contribs=normalized_hyp,
        outpath=base_out_path
    )
    pos_clusts = process_modisco_patterns(
        patterns=pos_patterns,
        df=screen_df,
        id_prefix="c"
    )
    neg_clusts = process_modisco_patterns(
        patterns=neg_patterns,
        df=screen_df,
        id_prefix="n"
    )
    return pos_clusts, neg_clusts


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute held-out DeepLIFT attributions across model folds and "
            "cluster them with TF-MoDISco."
        )
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=DEFAULT_BASE_CHECKPOINT_PATH,
        help=(
            "base checkpoint path; fold checkpoints are resolved by adding "
            "-fold-N before the suffix"
        )
    )
    parser.add_argument(
        "--outpath",
        type=Path,
        help=(
            "TF-MoDISco HDF5 output path (default: CHECKPOINT with "
            "-modisco.h5 appended to its stem)"
        )
    )
    parser.add_argument("--screen-name", default="r255x")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=10)
    return parser


def attr_main(
    argv: list[str] | None = None
) -> tuple[list[ModiscoCluster], list[ModiscoCluster]]:
    args = _build_arg_parser().parse_args(argv)
    outpath = args.outpath or args.checkpoint.with_name(
        f"{args.checkpoint.stem}-modisco.h5"
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)

    pos_clusters, neg_clusters = attr_kfolds(
        base_checkpoint_path=args.checkpoint,
        base_out_path=outpath,
        screen_name=args.screen_name,
        n_splits=args.n_splits,
        sequence_length=args.sequence_length
    )
    print(f"Saved TF-MoDISco results to {outpath}")
    print(
        f"Found {len(pos_clusters)} positive and "
        f"{len(neg_clusters)} negative clusters"
    )
    return pos_clusters, neg_clusters


if __name__ == "__main__":
    attr_main()
