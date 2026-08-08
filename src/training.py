import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import calculate_betabinom_loss
from .truth import (
    EmergeCNNPaths,
    EmergeDataset,
    frequency_tempered_weights,
    load_train_val,
    set_seed
)
from .model import ConvModelFramework, ModelSpec, build_model
from .metadata import get_model_config, get_training_config, get_data_config


TRAIN_BATCH_SIZE = 1024
VAL_BATCH_SIZE = 4096


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


def check_finite_gradients(model: torch.nn.Module) -> None:
    bad_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all().item():
            bad_params.append(name)
    if bad_params:
        names = ", ".join(bad_params)
        raise FloatingPointError(
            f"Non-finite gradients detected in: {names}"
        )

def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    max_batches = None,
    verbose = True,
    progress_label: str | None = None,
    progress_position: int = 0
) -> float:
    model.train()
    losses = []

    device = next(model.parameters()).device
    progress = tqdm(
        train_loader,
        desc=(
            f"[{progress_label}] Training epoch"
            if progress_label
            else "Training epoch"
        ),
        unit=" batches",
        position=progress_position,
        disable=not verbose
    )
    for step, batch in enumerate(progress, start=1):
        batch = {
            name: tensor.to(device)
            for name, tensor in batch.items()
        }

        optimizer.zero_grad()
        forward = model(batch["sequence"])
        loss = calculate_betabinom_loss(
            batch["k"],
            batch["n"],
            forward.get("pi"),
            forward["mu"],
            forward["phi"],
            sample_weight=batch.get("loss_weight")
        )
        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"Non-finite training loss at step {step}: {loss.item()}"
            )

        loss.backward()
        check_finite_gradients(model)
        optimizer.step()

        loss_weight = batch.get("loss_weight")
        loss_normalizer = (
            loss_weight.sum().item()
            if loss_weight is not None
            else batch["sequence"].shape[0]
        )
        losses.append({
            "loss": loss.item() * loss_normalizer,
            "length": loss_normalizer
        })
        progress.set_postfix(loss=f"{loss.item():.4f}")

        if max_batches is not None and step >= max_batches:
            break
    return (
        sum(item["loss"] for item in losses)
        / sum(item["length"] for item in losses)
    )

def val_one_epoch(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    verbose = True,
    progress_label: str | None = None,
    progress_position: int = 0
) -> float:
    model.eval()
    losses = []

    device = next(model.parameters()).device
    progress = tqdm(
        val_loader,
        desc=(
            f"[{progress_label}] Validating epoch"
            if progress_label
            else "Validating epoch"
        ),
        unit=" batches",
        position=progress_position,
        disable=not verbose
    )
    with torch.no_grad():
        for batch in progress:
            batch = {
                name: tensor.to(device)
                for name, tensor in batch.items()
            }

            forward = model(batch["sequence"])
            loss = calculate_betabinom_loss(
                batch["k"],
                batch["n"],
                forward.get("pi"),
                forward["mu"],
                forward["phi"]
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite validation loss: {loss.item()}"
                )

            losses.append({
                "loss": loss.item() * batch["sequence"].shape[0],
                "length": batch["sequence"].shape[0]
            })
            progress.set_postfix(loss=f"{loss.item():.4f}")

    return (
        sum(item["loss"] for item in losses)
        / sum(item["length"] for item in losses)
    )

def fit_model(
    model: torch.nn.Module,
    paths: EmergeCNNPaths,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    seed: int = 42,
    max_epochs: int = 1000,
    checkpoint_path: str = "data/best_model.pt",
    patience: int = 5,
    min_delta: float = 1e-4,
    tail_weight_power: float = 0.0,
    verbose: bool = True,
    progress_label: str | None = None,
    progress_position: int = 0
) -> list[dict]:
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            verbose=verbose,
            progress_label=progress_label,
            progress_position=progress_position
        )
        val_loss = val_one_epoch(
            model,
            val_loader,
            verbose=verbose,
            progress_label=progress_label,
            progress_position=progress_position
        )
        phi = torch.nn.functional.softplus(model.phi_raw).item()
        psi = 1 / (phi + 1)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "concentration": phi,
            "overdispersion": psi
        })

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            metadata = {
                "model_config": get_model_config(model),
                "training_config": get_training_config(
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    seed=seed,
                    max_epochs=max_epochs,
                    patience=patience,
                    min_delta=min_delta,
                    tail_weight_power=tail_weight_power,
                ),
                "data_config": get_data_config(
                    paths=paths,
                )
            }
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "history": history.copy(),
                "metadata": metadata,
                "concentration": phi,
                "overdispersion": psi,
                "torch_version": torch.__version__
                }, checkpoint_path
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=next(model.parameters()).device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return history

def train_model(
    model: ConvModelFramework,
    checkpoint_path: str,
    max_epochs: int = 10000,
    patience: int = 5,
    min_delta: float = 1e-4,
    seed: int = 42,
    tail_weight_power: float = 0.25
    verbose: bool = True,
    progress_label: str | None = None,
    progress_position: int = 0
) -> list[dict]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    paths = EmergeCNNPaths(screen_name="r255x")
    train_df, val_df = load_train_val(paths, seed=seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loss_weights = frequency_tempered_weights(
        train_df["mle"],
        power=tail_weight_power
    )
    train_data = EmergeDataset(
        df=train_df,
        loss_weights=train_loss_weights
    )
    val_data = EmergeDataset(df=val_df)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        generator=generator
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False
    )

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    training_history = fit_model(
        model=model,
        paths=paths,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        seed=seed,
        max_epochs=max_epochs,
        checkpoint_path=checkpoint_path,
        patience=patience,
        min_delta=min_delta,
        tail_weight_power=tail_weight_power,
        verbose=verbose,
        progress_label=progress_label,
        progress_position=progress_position
    )

    return training_history
