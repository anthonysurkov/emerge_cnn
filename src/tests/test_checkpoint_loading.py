import unittest

import torch

from src.metadata import get_model_config
from src.model import (
    ConvLayerSpec,
    ConvModelFramework,
    ModelSpec,
    SharedHeadSpec,
    SplitHeadSpec,
    build_model,
)
from src.blocks import (
    ConvStack,
    SplitHidden,
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
        model = build_model(ModelSpec(
            preset_id="test-split",
            conv_layers=(
                ConvLayerSpec(filters=64, kernel_size=6),
            ),
            heads=SplitHeadSpec(pi_hidden_size=32, mu_hidden_size=16),
        ))
        self.assert_checkpoint_round_trip(model)

    def test_reconstructs_two_layer_shared_hidden_model(self):
        model = build_model(ModelSpec(
            preset_id="test-two-layer-shared",
            conv_layers=(
                ConvLayerSpec(filters=16, kernel_size=3),
                ConvLayerSpec(filters=64, kernel_size=4),
            ),
            heads=SharedHeadSpec(hidden_size=32),
        ))
        self.assert_checkpoint_round_trip(model)

        config = get_model_config(model)
        self.assertEqual(config["conv_class"], ConvStack.__name__)
        self.assertEqual(
            config["convolution"],
            {"in_channels": 4, "layers": [[16, 3], [64, 4]]},
        )


if __name__ == "__main__":
    unittest.main()
