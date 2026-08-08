import unittest

import numpy as np

from src.truth import editing_strata, frequency_tempered_weights


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


if __name__ == "__main__":
    unittest.main()
