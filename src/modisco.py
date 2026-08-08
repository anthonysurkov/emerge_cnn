"""Forward-strand-only adapter for modisco-lite.

modisco-lite treats a seqlet and its reverse complement as equivalent during
neighbor search, fine-grained affinity calculation, and pattern alignment.
EMERGe sequences have a fixed experimental orientation, so only forward
representations may be compared.
"""

from contextlib import contextmanager
from threading import RLock
from typing import Iterator

import numpy as np
import sklearn.preprocessing
from numba import njit, prange

from modiscolite import affinitymat, aggregator, gapped_kmer, util
from modiscolite.tfmodisco import TFMoDISco


_PATCH_LOCK = RLock()


@njit
def _sparse_row_dot(data, indices, indptr, row_a, row_b):
    """Return the dot product between two rows of one CSR matrix."""
    a = indptr[row_a]
    b = indptr[row_b]
    dot = 0.0

    while a < indptr[row_a + 1] and b < indptr[row_b + 1]:
        column_a = indices[a]
        column_b = indices[b]

        if column_a == column_b:
            dot += data[a] * data[b]
            a += 1
            b += 1
        elif column_a < column_b:
            a += 1
        else:
            b += 1

    return dot


@njit(parallel=True)
def _nearest_forward_neighbors(data, indices, indptr, n_neighbors):
    """Find top neighbors using only forward-forward sparse dot products."""
    n_rows = len(indptr) - 1
    neighbors = np.empty((n_rows, n_neighbors), dtype=np.int32)
    similarities = np.empty((n_rows, n_neighbors), dtype=np.float64)

    for row_a in prange(n_rows):
        row_similarities = np.empty(n_rows, dtype=np.float64)
        for row_b in range(n_rows):
            row_similarities[row_b] = _sparse_row_dot(
                data,
                indices,
                indptr,
                row_a,
                row_b
            )

        order = np.argsort(-row_similarities, kind="mergesort")[:n_neighbors]
        neighbors[row_a] = order
        similarities[row_a] = row_similarities[order]

    return similarities, neighbors


def cosine_similarity_from_seqlets_forward(
    seqlets,
    n_neighbors,
    sign,
    topn=20,
    min_k=4,
    max_k=6,
    max_gap=15,
    max_len=15,
    max_entries=500,
    alphabet_size=4
):
    """Compute coarse seqlet neighbors without reverse complements."""
    del alphabet_size

    forward = gapped_kmer._seqlet_to_gkmers(
        seqlets,
        topn,
        min_k,
        max_k,
        max_gap,
        max_len,
        max_entries,
        True,
        sign
    )
    forward = sklearn.preprocessing.normalize(forward, norm="l2", axis=1)

    k = min(n_neighbors + 1, forward.shape[0])
    return _nearest_forward_neighbors(
        forward.data,
        forward.indices,
        forward.indptr,
        k
    )


def jaccard_from_seqlets_forward(
    seqlets,
    min_overlap,
    filter_seqlets=None,
    seqlet_neighbors=None
):
    """Compute fine affinity using forward tracks while retaining shifts."""
    forward, _ = util.get_2d_data_from_patterns(seqlets)

    if filter_seqlets is None:
        filter_seqlets = seqlets
        filter_forward = forward
    else:
        filter_forward, _ = util.get_2d_data_from_patterns(filter_seqlets)

    if seqlet_neighbors is None:
        seqlet_neighbors = np.tile(
            np.arange(len(filter_seqlets)),
            (len(seqlets), 1)
        )
    else:
        seqlet_neighbors = np.asarray(seqlet_neighbors)

    return affinitymat.jaccard(
        seqlet_neighbors=seqlet_neighbors,
        X=filter_forward,
        Y=forward,
        min_overlap=min_overlap,
        func=int,
        return_sparse=True
    )


def align_patterns_forward(
    parent_pattern,
    child_pattern,
    metric,
    min_overlap,
    transformer,
    include_hypothetical
):
    """Find the best positional shift without considering strand reversal."""
    parent_forward, _ = util.get_2d_data_from_patterns(
        [parent_pattern],
        transformer=transformer,
        include_hypothetical=include_hypothetical
    )
    child_forward, _ = util.get_2d_data_from_patterns(
        [child_pattern],
        transformer=transformer,
        include_hypothetical=include_hypothetical
    )

    best_score, best_offset = metric(
        child_forward,
        parent_forward,
        min_overlap
    ).squeeze()

    return int(best_offset), False, best_score


@contextmanager
def _forward_only_algorithms() -> Iterator[None]:
    """Temporarily install forward-only algorithms in modisco-lite."""
    with _PATCH_LOCK:
        original_cosine = affinitymat.cosine_similarity_from_seqlets
        original_jaccard = affinitymat.jaccard_from_seqlets
        original_aligner = aggregator._align_patterns

        affinitymat.cosine_similarity_from_seqlets = (
            cosine_similarity_from_seqlets_forward
        )
        affinitymat.jaccard_from_seqlets = jaccard_from_seqlets_forward
        aggregator._align_patterns = align_patterns_forward

        try:
            yield
        finally:
            affinitymat.cosine_similarity_from_seqlets = original_cosine
            affinitymat.jaccard_from_seqlets = original_jaccard
            aggregator._align_patterns = original_aligner


def _reject_reverse_complements(patterns) -> None:
    if patterns is None:
        return

    reverse_seqlets = [
        seqlet
        for pattern in patterns
        for seqlet in pattern.seqlets
        if seqlet.is_revcomp
    ]
    if reverse_seqlets:
        raise RuntimeError(
            "Forward-only TF-MoDISco produced "
            f"{len(reverse_seqlets)} reverse-complemented seqlets"
        )


def tfmodisco_forward_only(*args, **kwargs):
    """Run TF-MoDISco without reverse-complement matching or alignment."""
    with _forward_only_algorithms():
        pos_patterns, neg_patterns = TFMoDISco(*args, **kwargs)

    _reject_reverse_complements(pos_patterns)
    _reject_reverse_complements(neg_patterns)
    return pos_patterns, neg_patterns
