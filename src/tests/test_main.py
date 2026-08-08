import unittest
from pathlib import Path
from unittest.mock import patch

from src.main import scan_conv2_sharedmlp
from src.model import config_conv2_sharedmlp
from src.training import _train_spec, _train_specs_in_subprocesses


class MainScanTests(unittest.TestCase):
    @patch("src.training.tqdm.write")
    @patch("src.training.set_seed")
    @patch("src.training.build_model")
    @patch("src.training.train_model")
    def test_worker_labels_terminal_progress(
        self,
        train_model,
        build_model,
        set_seed,
        tqdm_write
    ):
        train_model.return_value = [{"epoch": 1}, {"epoch": 2}]
        spec = config_conv2_sharedmlp(32, 3, 128, 2, 32)

        epochs = _train_spec(spec, Path("model.pt"), 0.5, agent_id=7)

        self.assertEqual(epochs, 2)
        set_seed.assert_called_once_with()
        train_model.assert_called_once_with(
            model=build_model.return_value,
            checkpoint_path=Path("model.pt"),
            tail_weight_power=0.5,
            progress_label="agent 7",
            progress_position=6
        )
        self.assertEqual(tqdm_write.call_count, 2)

    @patch("src.main.eval_main")
    @patch("src.main._train_specs_in_subprocesses")
    def test_trains_and_evaluates_full_parameter_grid(
        self,
        train_specs,
        eval_main
    ):
        train_specs.return_value = [1] * 8
        eval_main.return_value = {"val_loss": 0.5}

        with patch("src.main.DATA_DIR", Path("/tmp/cnn-main-test")):
            results = scan_conv2_sharedmlp(
                f1_range=[32],
                k1_range=[3, 4],
                f2_range=[128],
                k2_range=[2, 3, 4, 5],
                h_range=[32],
                tail_weight_power=0.5,
                max_workers=3
            )

        self.assertEqual(len(results), 8)
        train_specs.assert_called_once()
        self.assertEqual(eval_main.call_count, 8)
        self.assertEqual(len({result["preset_id"] for result in results}), 8)
        self.assertTrue(all(result["epochs_trained"] == 1 for result in results))
        self.assertEqual(train_specs.call_args.kwargs["tail_weight_power"], 0.5)
        self.assertEqual(train_specs.call_args.kwargs["max_workers"], 3)
        self.assertEqual(len(train_specs.call_args.kwargs["specs"]), 8)
        for path in train_specs.call_args.kwargs["checkpoint_paths"]:
            self.assertEqual(path.suffix, ".pt")

    def test_rejects_nonpositive_worker_count(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _train_specs_in_subprocesses([], [], 0.25, max_workers=0)


if __name__ == "__main__":
    unittest.main()
