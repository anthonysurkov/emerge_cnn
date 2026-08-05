"""Collapse the full k-mer coefficient vector into a position x base logo.

The top-coefficient list fragments one degenerate motif across many exact 6-mers.
This refits the selected model, saves the full coefficient vector, and aggregates
it into a 10 x 4 position/base importance map so the shared core is explicit.

Writes only NEW files under the results dir; it never overwrites the original
``metrics.json`` or ``motif_coefficients.csv``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .featurize import ALPHABET, build_design, encode


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "r255x.csv"
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "results"


def aggregate_position_base(coef, blocks, length):
    """Sum coefficients by (position, base) across all widths, and for k=1 only.

    ``all_k`` attributes every feature's weight to each (position, base) it
    fixes, concentrating the shared-core signal that the 6-mer list scatters.
    ``k1`` is the pure additive positional effect (one feature per position).
    """
    all_k = np.zeros((length, len(ALPHABET)))
    k1 = np.zeros((length, len(ALPHABET)))
    for k, start, offset in blocks:
        block = coef[offset:offset + 4 ** k]
        codes = np.arange(4 ** k)
        for j in range(k):
            base_at_j = (codes // (4 ** j)) % 4
            sums = np.bincount(base_at_j, weights=block, minlength=len(ALPHABET))
            all_k[start + j] += sums
            if k == 1:
                k1[start + j] += sums
    return all_k, k1


def _plot_logo(all_k, k1, length, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = np.arange(1, length + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, matrix, title in (
        (axes[0], k1, "k=1 additive positional effect"),
        (axes[1], all_k, "all-width aggregated importance"),
    ):
        limit = np.abs(matrix).max() or 1.0
        # Diverging map centered at zero: cool (down) - neutral - warm (up).
        image = ax.imshow(
            matrix.T, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto"
        )
        ax.set_xticks(range(length))
        ax.set_xticklabels(positions)
        ax.set_yticks(range(len(ALPHABET)))
        ax.set_yticklabels(ALPHABET)
        ax.set_xlabel("position (1-based)")
        ax.set_title(title)
        # Per-cell numeric labels so the map reads without relying on color.
        for p in range(length):
            for b in range(len(ALPHABET)):
                value = matrix[p, b]
                ax.text(p, b, f"{value:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="black" if abs(value) < 0.6 * limit else "white")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                     label="signed coefficient (+ raises editing)")
    fig.suptitle("k-mer regression motif map (editing rate), positions 5-10 core")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--kmax", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=100.0,
                        help="regularization strength (the value the search selected)")
    parser.add_argument("--seq-column", default="5to3")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    table = pd.read_csv(args.csv, usecols=[args.seq_column, "n", "mle"])
    codes = encode(table[args.seq_column].to_numpy())
    length = codes.shape[1]
    design, blocks = build_design(codes, kmax=args.kmax)
    print(f"design {design.shape}, refitting ridge alpha={args.alpha} on all "
          f"{design.shape[0]} sequences...", flush=True)

    model = Ridge(alpha=args.alpha, fit_intercept=True)
    model.fit(design, table["mle"].to_numpy(), sample_weight=table["n"].to_numpy())
    print(f"fit done ({time.time() - started:.1f}s)", flush=True)

    # Full coefficient vector + block index (new file; originals preserved).
    np.savez(
        outdir / "full_coefficients.npz",
        coef=model.coef_, intercept=model.intercept_,
        blocks=np.array(blocks, dtype=int), alpha=args.alpha,
    )
    all_k, k1 = aggregate_position_base(model.coef_, blocks, length)

    rows = []
    for p in range(length):
        for b, base in enumerate(ALPHABET):
            rows.append({
                "position": p + 1, "base": base,
                "k1_coefficient": float(k1[p, b]),
                "all_width_importance": float(all_k[p, b]),
            })
    importance = pd.DataFrame(rows)
    importance.to_csv(outdir / "position_base_importance.csv", index=False)
    _plot_logo(all_k, k1, length, outdir / "position_base_logo.png")

    print("\naggregated all-width importance (rows=base, cols=position 1-10):")
    frame = pd.DataFrame(all_k.T, index=list(ALPHABET),
                         columns=range(1, length + 1)).round(3)
    print(frame.to_string())
    print(f"\nWrote full_coefficients.npz, position_base_importance.csv, "
          f"position_base_logo.png under {outdir}")
    print(f"(originals metrics.json / motif_coefficients.csv untouched; "
          f"{time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
