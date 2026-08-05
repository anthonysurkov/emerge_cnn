import unittest

import numpy as np

from emerge_umap.kmer.featurize import build_design, encode, feature_name, feature_names
from emerge_umap.kmer.regression import fit_kmer_regression


class FeaturizeTests(unittest.TestCase):
    def test_encode_maps_u_and_t_together(self):
        codes = encode(["ACGT", "ACGU"])
        np.testing.assert_array_equal(codes[0], codes[1])
        np.testing.assert_array_equal(codes[0], [0, 1, 2, 3])

    def test_encode_rejects_ragged_and_invalid(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            encode(["ACG", "ACGT"])
        with self.assertRaisesRegex(ValueError, "outside"):
            encode(["ACGX"])

    def test_design_shape_and_row_sparsity(self):
        codes = encode(["ACGTAC", "TTTTTT"])
        design, blocks = build_design(codes, kmax=3)
        # widths 1..3 on length 6 -> (6)+(5)+(4) = 15 nonzeros per row
        self.assertEqual(design.shape, (2, 6 * 4 + 5 * 16 + 4 * 64))
        np.testing.assert_array_equal(design.sum(axis=1).A1, [15, 15])

    def test_feature_name_round_trips_a_known_column(self):
        codes = encode(["ACGT"])
        design, blocks = build_design(codes, kmax=2)
        names = feature_names(blocks)
        # exactly one k=1 column is active at each of the 4 positions
        active = np.flatnonzero(design.toarray()[0])
        decoded = [feature_name(int(c), blocks) for c in active]
        self.assertIn((1, 1, "A"), decoded)
        self.assertIn((1, 4, "T"), decoded)
        self.assertEqual(len(names), design.shape[1])

    def test_regression_recovers_a_planted_position_effect(self):
        # Sequences with A at position 1 edit hard; everything else is noise.
        rng = np.random.default_rng(0)
        bases = np.array(list("ACGT"))
        seqs, rate, cover = [], [], []
        for _ in range(2000):
            s = "".join(rng.choice(bases, size=5))
            seqs.append(s)
            rate.append(0.6 if s[0] == "A" else 0.02)
            cover.append(200)
        codes = encode(seqs)
        design, blocks = build_design(codes, kmax=2)
        result = fit_kmer_regression(
            design, np.array(rate), np.array(cover), blocks,
            alphas=(0.01, 0.1, 1.0), seed=0,
        )
        self.assertGreater(result.metrics["test_weighted_r2"], 0.8)
        top = result.coefficients.iloc[0]
        self.assertEqual((top["k"], top["position"], top["kmer"]), (1, 1, "A"))


if __name__ == "__main__":
    unittest.main()
