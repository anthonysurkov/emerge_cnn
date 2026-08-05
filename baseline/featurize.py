"""Position-aware k-mer feature construction for fixed-length sequences.

Every (width, position, exact k-mer) combination becomes one binary column, for
all widths ``1..kmax`` at once. The caller does not pick a single ``k``: the full
hierarchy is emitted and a regularized regression selects which widths carry
signal. Design matrices are sparse (each row has ``sum_k (L-k+1)`` nonzeros).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


ALPHABET = ("A", "C", "G", "T")
# RNA U and DNA T share a code so either input alphabet works.
_ASCII_TO_CODE = np.full(256, -1, dtype=np.int64)
for _base, _code in zip("ACGTU", (0, 1, 2, 3, 3)):
    _ASCII_TO_CODE[ord(_base)] = _code


def encode(sequences) -> np.ndarray:
    """Encode equal-length sequences as an (n, L) array of base codes 0-3."""
    values = np.char.upper(np.asarray(sequences, dtype=str))
    lengths = np.char.str_len(values)
    if len(values) == 0:
        raise ValueError("at least one sequence is required")
    if np.any(lengths != lengths[0]):
        raise ValueError("all sequences must have the same length")
    length = int(lengths[0])
    flat = np.frombuffer("".join(values).encode("ascii"), dtype=np.uint8)
    codes = _ASCII_TO_CODE[flat].reshape(len(values), length)
    if np.any(codes < 0):
        raise ValueError("sequences contain symbols outside A/C/G/T/U")
    return codes


def build_design(codes: np.ndarray, kmax: int = 6):
    """Build the sparse position-aware k-mer design matrix for widths 1..kmax.

    Returns the CSR matrix and a list of ``(k, start, offset)`` blocks that
    :func:`feature_name` uses to decode a column index back to its motif.
    """
    if kmax < 1:
        raise ValueError("kmax must be positive")
    n, length = codes.shape
    if kmax > length:
        raise ValueError("kmax cannot exceed the sequence length")

    row_index = np.arange(n)
    blocks, columns = [], []
    offset = 0
    for k in range(1, kmax + 1):
        powers = (4 ** np.arange(k)).astype(np.int64)
        for start in range(length - k + 1):
            window = codes[:, start:start + k].astype(np.int64)
            columns.append(offset + window @ powers)
            blocks.append((k, start, offset))
            offset += 4 ** k

    rows = np.repeat(row_index, len(blocks))
    cols = np.stack(columns, axis=1).reshape(-1)
    data = np.ones(len(cols), dtype=np.float32)
    design = sparse.csr_matrix((data, (rows, cols)), shape=(n, offset))
    return design, blocks


def _decode_kmer(code: int, k: int) -> str:
    bases = []
    for _ in range(k):
        bases.append(ALPHABET[code % 4])
        code //= 4
    return "".join(bases)


def feature_name(column: int, blocks) -> tuple[int, int, str]:
    """Decode a column index to ``(k, one_based_position, kmer)``."""
    for k, start, offset in blocks:
        if offset <= column < offset + 4 ** k:
            return k, start + 1, _decode_kmer(column - offset, k)
    raise IndexError(f"column {column} is out of range")


def feature_names(blocks) -> np.ndarray:
    """All feature names as an object array aligned with design columns."""
    names = []
    for k, start, offset in blocks:
        for code in range(4 ** k):
            names.append((k, start + 1, _decode_kmer(code, k)))
    return np.array(names, dtype=object)
