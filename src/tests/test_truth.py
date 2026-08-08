import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.truth import editing_strata, frequency_tempered_weights, make_splits


class FrequencyTemperedWeightTests(unittest.TestCase):
    def test_editing_strata_include_distinct_extreme_tail(self):
        mle = np.array([0.0, 0.01, 0.02, 0.05, 0.10, 0.30, 0.40, 0.50,
                        0.64, 0.80])

        np.testing.assert_array_equal(
            editing_strata(mle),
            np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
        )

    def test_zero_power_disables_weighting(self):
        weights = frequency_tempered_weights(
            np.array([0.0, 0.0, 0.01, 0.50, 0.80]),
            power=0.0
        )

        np.testing.assert_array_equal(weights, np.ones(5, dtype=np.float32))

    def test_rare_strata_receive_larger_mean_one_weights(self):
        mle = np.array([0.0] * 16 + [0.01] * 4 + [0.80])
        weights = frequency_tempered_weights(mle, power=0.25)

        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(weights[-1], weights[16])
        self.assertGreater(weights[16], weights[0])
        self.assertAlmostEqual(
            float(weights[-1] / weights[0]),
            16 ** 0.25,
            places=6
        )

    def test_power_must_be_between_zero_and_one(self):
        for power in (-0.1, 1.1):
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                frequency_tempered_weights(np.array([0.0]), power=power)


class SplitTests(unittest.TestCase):
    def test_split_classes_use_k_zero_not_mle_cutoff(self):
        df = pd.DataFrame({
            "k": [1, 2, 0, 0],
            "mle": [0.01, 0.20, 0.0, 0.0],
        })
        split_inputs = []

        def fake_train_test_split(
            frame,
            *,
            test_size,
            stratify,
            random_state
        ):
            split_inputs.append(frame.copy())
            cut = int(len(frame) * (1.0 - test_size))
            return frame.iloc[:cut], frame.iloc[cut:]

        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            paths = SimpleNamespace(
                train_path=directory / "train.csv",
                val_path=directory / "val.csv",
                test_path=directory / "test.csv",
            )
            with patch(
                "src.truth.train_test_split",
                side_effect=fake_train_test_split
            ):
                make_splits(df, paths, force_regenerate=True)

        self.assertEqual(set(split_inputs[0].index), {0, 1})
        self.assertEqual(set(split_inputs[2].index), {2, 3})
        self.assertIn(0, split_inputs[0].index)


if __name__ == "__main__":
    unittest.main()
