"""Position-aware k-mer regression of editing rate (interpretable baseline)."""

from .featurize import build_design, encode, feature_name, feature_names
from .regression import KmerRegression, fit_kmer_regression

__all__ = [
    "encode", "build_design", "feature_name", "feature_names",
    "KmerRegression", "fit_kmer_regression",
]
