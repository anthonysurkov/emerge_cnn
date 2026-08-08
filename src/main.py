from pprint import pprint
from typing import Any

from .truth import set_seed
from .training import TAIL_WEIGHT_POWER, _train_specs_in_subprocesses
from .model import scanconfig_conv2_sharedmlp
from .eval import eval_main
from .paths import DATA_DIR

def scan_conv2_sharedmlp(
    f1_range: list[int],
    k1_range: list[int],
    f2_range: list[int],
    k2_range: list[int],
    h_range: list[int],
    *,
    tail_weight_power: float = TAIL_WEIGHT_POWER,
    max_workers: int = 2
) -> list[dict[str, Any]]:
    specs = scanconfig_conv2_sharedmlp(
        f1_range,
        k1_range,
        f2_range,
        k2_range,
        h_range
    )
    checkpoint_paths = [
        DATA_DIR / f"{spec.preset_id}.pt"
        for spec in specs
    ]
    epochs_trained = _train_specs_in_subprocesses(
        specs=specs,
        checkpoint_paths=checkpoint_paths,
        tail_weight_power=tail_weight_power,
        max_workers=max_workers
    )

    eval_results = []
    for spec, checkpoint_path, epochs in zip(
        specs,
        checkpoint_paths,
        epochs_trained
    ):
        statistics = dict(eval_main(checkpoint_path=checkpoint_path))
        statistics["preset_id"] = spec.preset_id
        statistics["epochs_trained"] = epochs
        eval_results.append(statistics)

    return eval_results


def main():
    set_seed()
    results = scan_conv2_sharedmlp(
        f1_range=[32],
        k1_range=[3, 4],
        f2_range=[128],
        k2_range=[2, 3, 4, 5],
        h_range=[32],
        tail_weight_power=0.25,
        max_workers=2
    )
    pprint(results)


if __name__ == "__main__":
    main()
