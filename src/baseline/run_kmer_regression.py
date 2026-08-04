"""Fit a position-aware k-mer regression of editing rate over the full library.

Loads the r255x N10 table, builds position-aware 1..kmax-mer features, fits a
coverage-weighted regularized regression, and writes held-out metrics plus the
signed motif map (top editing-increasing and editing-decreasing k-mers).

Example::

    python3 -m emerge_umap.kmer.run_kmer_regression
    python3 -m emerge_umap.kmer.run_kmer_regression --penalty lasso --kmax 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .featurize import build_design, encode
from .regression import fit_kmer_regression


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "r255x.csv"
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--kmax", type=int, default=6)
    parser.add_argument("--penalty", choices=("ridge", "lasso", "elasticnet"), default="ridge")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None, help="subsample for a quick run")
    parser.add_argument("--seq-column", default="5to3")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    table = pd.read_csv(args.csv, usecols=[args.seq_column, "n", "k", "mle"])
    if args.max_rows:
        table = table.sample(n=min(args.max_rows, len(table)), random_state=args.seed)
    table = table.reset_index(drop=True)
    print(f"loaded {len(table)} sequences in {time.time() - started:.1f}s")

    codes = encode(table[args.seq_column].to_numpy())
    design, blocks = build_design(codes, kmax=args.kmax)
    print(f"design matrix: {design.shape[0]} x {design.shape[1]} "
          f"({design.nnz} nonzeros)")

    print("fitting (alpha search)...", flush=True)
    result = fit_kmer_regression(
        design, table["mle"].to_numpy(), table["n"].to_numpy(), blocks,
        penalty=args.penalty, alphas=tuple(args.alphas), l1_ratio=args.l1_ratio,
        test_size=args.test_size, seed=args.seed, verbose=True,
    )

    (outdir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
    result.coefficients.to_csv(outdir / "motif_coefficients.csv", index=False)

    print("\n== held-out metrics ==")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")
    print("\n== top editing-increasing k-mers ==")
    print(result.coefficients.head(15).to_string(index=False))
    print("\n== top editing-decreasing k-mers ==")
    print(result.coefficients.tail(15).to_string(index=False))
    print(f"\nSaved outputs under {outdir} ({time.time() - started:.1f}s total)")


if __name__ == "__main__":
    main()
