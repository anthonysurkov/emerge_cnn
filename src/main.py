import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from pprint import pprint
from typing import Any

from tqdm import tqdm

from .truth import set_seed
from .training import TAIL_WEIGHT_POWER, train_model
from .model import ModelSpec, scanconfig_conv2_sharedmlp, build_model
from .eval import eval_main
from .paths import DATA_DIR


def _configure_training_worker(progress_lock) -> None:
    tqdm.set_lock(progress_lock)

def _train_spec(
    spec: ModelSpec,
    checkpoint_path: Path,
    tail_weight_power: float,
    agent_id: int
) -> int:
    label = f"agent {agent_id}"
    tqdm.write(f"[{label}] starting {spec.preset_id}")
    set_seed()
    training_history = train_model(
        model=build_model(spec=spec),
        checkpoint_path=checkpoint_path,
        tail_weight_power=tail_weight_power,
        progress_label=label,
        progress_position=agent_id - 1
    )
    tqdm.write(
        f"[{label}] finished {spec.preset_id} "
        f"after {len(training_history)} epochs"
    )
    return len(training_history)

def _train_specs_in_subprocesses(
    specs: list[ModelSpec],
    checkpoint_paths: list[Path],
    tail_weight_power: float,
    max_workers: int
) -> list[int]:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")

    context = multiprocessing.get_context("spawn")
    progress_lock = context.RLock()
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=_configure_training_worker,
        initargs=(progress_lock,)
    ) as executor:
        futures = [
            executor.submit(
                _train_spec,
                spec,
                checkpoint_path,
                tail_weight_power,
                agent_id
            )
            for agent_id, (spec, checkpoint_path) in enumerate(
                zip(specs, checkpoint_paths),
                start=1
            )
        ]
        return [future.result() for future in futures]

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
