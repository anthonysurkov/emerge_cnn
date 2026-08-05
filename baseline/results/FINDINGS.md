# k-mer regression findings (r255x, N10 editing)

Position-aware k-mer ridge regression of editing rate over the full library
(889,791 unique N10 sequences, coverage-weighted, binomial-motivated).

## Predictive result

- Held-out **weighted R² = 0.547** (unweighted 0.494; Pearson 0.704, 0.745 on the
  editable tail). Additive positional k-mer features explain ~55% of
  coverage-weighted editing variance. Signal exists, decisively.
- Insensitive to regularization: validation R² = 0.562 across alpha in
  {0.1, 1, 10, 100}; the search selected alpha=100 (most shrunk, equally good).
- Top coefficients are extreme outliers: coef std = 0.0043, and the top 6-mer
  (+0.157) is 35.5 sigma out, the single largest of 29,112 features. In absolute
  terms +0.16 editing rate from one 6-mer (~25x the library mean of 0.005).

## The motif is a gapped / dyad motif (weak-vs-strong marginal structure)

Coverage-weighted mean editing by (position, base) — model-free, so not a ridge
artifact — shows the informative positions are **5, 6, 7 and 10, with 8-9 a
relative gap** and 1-4 flat background. Full table:
`marginal_editing_by_position.csv`.

| position | strength | top base | contrast |
|---|---|---|---|
| 1-4 | weak/gap (background) | - | 0.0010-0.0017 |
| 5 | strong | T | 0.0034 |
| 6 | strong | T | 0.0039 |
| 7 | moderate | C | 0.0027 |
| 8 | weak/gap | A | 0.0015 |
| 9 | weak/gap | C | 0.0015 |
| 10 | strong | C/T (pyrimidine up, purine down) | 0.0038 |

Contrast = (max - min) coverage-weighted mean editing across the four bases at
that position; library mean editing = 0.0051.

So the motif is roughly a **TC-rich core at 5-7, a near-free gap at 8-9, and a
decisive terminal pyrimidine at 10** (a dyad / spaced motif).

## Why this is the wrong object

The top signal is width-6, but those 6-mers all share a degenerate ~3-mer core
plus a variable middle (`TCC-AT-T`, `TCC-GT-T`, `TCC-GG-T`, `TCC-AA-T`, ...).
Contiguous exact-k-mer features **cannot express a don't-care position**, so one
gapped motif fragments across all fillings of the 8-9 gap. This is a
representation mismatch, not a tuning problem; larger k makes it worse.

The right object is **distributional over positions** (a PWM, where a gap is a
low-information column) or **learned filters** (a CNN, where a gap is near-zero
weight), not exact-match k-mers.

Note: the same failure mode is a milder limitation of emerge_umap's own motif
window (contiguous, capped at 6 nt). Here the 5-10 informative span is exactly
6 nt so a single window just covers it, but a wider dyad or two separated windows
would clip — a limitation PIPELINE.md already lists.

## Cross-check against emerge_umap clusters

9 of 12 emerge_umap final clusters (built sequence-only, editing held out) have
their positions-5-10 consensus among the regression's top-20 editing-increasing
6-mers; the other 3 are one base off. All 12 clusters have mean editing 0.41-0.55
vs the library mean 0.006. Two methods with opposite supervision converge on the
same motifs -> independent validation.

## Files

- `metrics.json`, `motif_coefficients.csv` — regression metrics and top k-mers.
- `marginal_editing_by_position.csv` — the weak/strong per-position map above.
- `full_coefficients.npz`, `position_base_importance.csv`,
  `position_base_logo.png` — full coefficients and aggregated logo. NOTE: the
  "all-width aggregated" logo panel is an invalid summary (summing over the ~1000
  noise k-mers per position inverts the sign vs the k=1 panel and the marginal
  map); use the k=1 panel or the marginal map, not the aggregate.
