import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.attr import do_modisco


class ModiscoInputTests(unittest.TestCase):
    @patch("src.attr.save_hdf5")
    @patch("src.attr.tfmodisco_forward_only", return_value=(None, None))
    def test_adds_neutral_padding(
        self,
        mock_modisco,
        mock_save_hdf5
    ):
        ohe = torch.zeros((2, 4, 10), dtype=torch.float32)
        ohe[:, 0, :] = 1.0
        hypothetical = torch.arange(
            80,
            dtype=torch.float32
        ).reshape(2, 4, 10)

        do_modisco(
            ohe=ohe,
            hyp_contribs=hypothetical,
            outpath="test.h5",
            window_size=6,
            flank_size=2
        )

        call = mock_modisco.call_args.kwargs
        padded_ohe = call["one_hot"]
        padded_hypothetical = call["hypothetical_contribs"]

        self.assertEqual(padded_ohe.shape, (2, 14, 4))
        self.assertEqual(padded_hypothetical.shape, (2, 14, 4))
        np.testing.assert_array_equal(
            padded_ohe[:, 2:-2, :],
            ohe.permute(0, 2, 1).numpy()
        )
        np.testing.assert_array_equal(
            padded_hypothetical[:, 2:-2, :],
            hypothetical.permute(0, 2, 1).numpy()
        )
        np.testing.assert_array_equal(padded_ohe[:, :2, :], 0.25)
        np.testing.assert_array_equal(padded_ohe[:, -2:, :], 0.25)
        np.testing.assert_array_equal(padded_hypothetical[:, :2, :], 0.0)
        np.testing.assert_array_equal(padded_hypothetical[:, -2:, :], 0.0)
        self.assertEqual(call["flank_size"], 2)
        mock_save_hdf5.assert_called_once_with(
            "test.h5",
            None,
            None,
            window_size=10
        )

    def test_rejects_zero_flank(self):
        inputs = torch.zeros((1, 4, 10))

        with self.assertRaisesRegex(ValueError, "at least 1"):
            do_modisco(
                ohe=inputs,
                hyp_contribs=inputs,
                flank_size=0
            )


if __name__ == "__main__":
    unittest.main()
