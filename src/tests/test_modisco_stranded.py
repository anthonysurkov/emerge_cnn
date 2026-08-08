import unittest
from unittest.mock import patch

import numpy as np

from modiscolite import affinitymat, aggregator
from modiscolite.core import Seqlet, SeqletSet

from src.modisco_stranded import (
    align_patterns_forward,
    cosine_similarity_from_seqlets_forward,
    jaccard_from_seqlets_forward,
    tfmodisco_forward_only,
)


BASE_TO_INDEX = {base: index for index, base in enumerate("ACGT")}


def make_seqlet(sequence: str, example_idx: int = 0) -> Seqlet:
    seqlet = Seqlet(
        example_idx=example_idx,
        start=0,
        end=len(sequence),
        is_revcomp=False
    )
    one_hot = np.eye(4)[
        [BASE_TO_INDEX[base] for base in sequence]
    ]
    seqlet.sequence = one_hot
    seqlet.contrib_scores = one_hot.copy()
    seqlet.hypothetical_contribs = one_hot.copy()
    return seqlet


class ForwardOnlySimilarityTests(unittest.TestCase):
    def setUp(self):
        self.forward = make_seqlet("AAAAAACCCC", example_idx=0)
        self.reverse_complement = make_seqlet("GGGGTTTTTT", example_idx=1)
        self.forward_neighbor = make_seqlet("AAAAAAGCCC", example_idx=2)
        self.seqlets = [
            self.forward,
            self.reverse_complement,
            self.forward_neighbor,
        ]

    def test_coarse_neighbors_do_not_equate_reverse_complements(self):
        similarities, neighbors = cosine_similarity_from_seqlets_forward(
            self.seqlets,
            n_neighbors=1,
            sign=1
        )

        self.assertEqual(neighbors[0].tolist(), [0, 2])
        self.assertLess(similarities[0, 1], 1.0)

    def test_fine_affinity_does_not_equate_reverse_complements(self):
        neighbors = np.array([[1], [0], [0]])

        stranded = jaccard_from_seqlets_forward(
            self.seqlets,
            min_overlap=1.0,
            seqlet_neighbors=neighbors
        )
        unstranded = affinitymat.jaccard_from_seqlets(
            self.seqlets,
            min_overlap=1.0,
            seqlet_neighbors=neighbors
        )

        self.assertEqual(stranded[0, 0], 0.0)
        self.assertEqual(unstranded[0, 0], 1.0)

    def test_pattern_alignment_never_reverse_complements(self):
        offset, is_revcomp, score = align_patterns_forward(
            self.forward,
            self.reverse_complement,
            metric=affinitymat.pearson_correlation,
            min_overlap=1.0,
            transformer="magnitude",
            include_hypothetical=False
        )

        self.assertEqual(offset, 0)
        self.assertFalse(is_revcomp)
        self.assertLess(score, 0.0)

    def test_pattern_alignment_still_finds_positional_shift(self):
        parent = make_seqlet("AACCGGTTAA")
        child = make_seqlet("CCGGTT")

        offset, is_revcomp, score = align_patterns_forward(
            parent,
            child,
            metric=affinitymat.pearson_correlation,
            min_overlap=0.5,
            transformer="magnitude",
            include_hypothetical=False
        )

        self.assertEqual(offset, 2)
        self.assertFalse(is_revcomp)
        self.assertEqual(score, 1.0)


class ForwardOnlyAdapterTests(unittest.TestCase):
    def test_installs_all_three_algorithms_only_during_call(self):
        original_cosine = affinitymat.cosine_similarity_from_seqlets
        original_jaccard = affinitymat.jaccard_from_seqlets
        original_aligner = aggregator._align_patterns

        def inspect_algorithms(*args, **kwargs):
            del args, kwargs
            self.assertIs(
                affinitymat.cosine_similarity_from_seqlets,
                cosine_similarity_from_seqlets_forward
            )
            self.assertIs(
                affinitymat.jaccard_from_seqlets,
                jaccard_from_seqlets_forward
            )
            self.assertIs(aggregator._align_patterns, align_patterns_forward)
            return [], []

        with patch(
            "src.modisco_stranded.TFMoDISco",
            side_effect=inspect_algorithms
        ):
            self.assertEqual(tfmodisco_forward_only(), ([], []))

        self.assertIs(
            affinitymat.cosine_similarity_from_seqlets,
            original_cosine
        )
        self.assertIs(affinitymat.jaccard_from_seqlets, original_jaccard)
        self.assertIs(aggregator._align_patterns, original_aligner)

    def test_rejects_reverse_complemented_output(self):
        reverse_seqlet = make_seqlet("ACGT")
        reverse_seqlet.is_revcomp = True
        patterns = [SeqletSet([reverse_seqlet])]

        with patch(
            "src.modisco_stranded.TFMoDISco",
            return_value=(patterns, None)
        ):
            with self.assertRaisesRegex(RuntimeError, "reverse-complemented"):
                tfmodisco_forward_only()


if __name__ == "__main__":
    unittest.main()
