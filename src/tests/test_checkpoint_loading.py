import unittest

import torch

from src.metadata import get_model_config
from src.model import (
    ConvModelFramework,
    OneHotFeats,
    OneLayerConv,
    SplitHidden,
    TwoLayerConv,
    SharedHidden,
)
from src.utils import model_from_checkpoint


class CheckpointLoadingTests(unittest.TestCase):
    def assert_checkpoint_round_trip(self, expected: ConvModelFramework):
        checkpoint = {
            "metadata": {"model_config": get_model_config(expected)},
            "model_state_dict": expected.state_dict(),
        }

        actual = model_from_checkpoint(checkpoint)

        self.assertIs(type(actual), type(expected))
        self.assertIs(type(actual.encoder_block), type(expected.encoder_block))
        self.assertIs(type(actual.conv_block), type(expected.conv_block))
        self.assertIs(type(actual.heads_block), type(expected.heads_block))
        for name, tensor in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[name], tensor)

    def test_reconstructs_split_hidden_model(self):
        self.assert_checkpoint_round_trip(ConvModelFramework(
            encoder_block=OneHotFeats(),
            conv_block=OneLayerConv(num_filters=64, kernel_size=6),
            heads_block=SplitHidden(
                input_size=320,
                hidden_size_pi=32,
                hidden_size_mu=16,
            ),
            phi_init=1,
        ))

    def test_reconstructs_two_layer_shared_hidden_model(self):
        self.assert_checkpoint_round_trip(ConvModelFramework(
            encoder_block=OneHotFeats(),
            conv_block=TwoLayerConv(
                num_filters_layer_one=16,
                num_filters_layer_two=64,
                kernel_size_layer_one=3,
                kernel_size_layer_two=4,
            ),
            heads_block=SharedHidden(input_size=320, hidden_size=32),
            phi_init=1,
        ))


if __name__ == "__main__":
    unittest.main()
