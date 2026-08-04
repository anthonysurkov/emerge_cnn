import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from core import calculate_loss


def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    max_batches = None,
    verbose = True
) -> float:
    model.train()
    losses = []

    progress = tqdm(
        train_loader,
        desc="Training epoch",
        unit=" batches",
        disable=not verbose
    )
    for step, batch in enumerate(progress, start=1):
        optimizer.zero_grad()
        forward = model(batch["sequence"])
        loss = calculate_loss(
            batch["k"],
            batch["n"],
            forward["pi"],
            forward["mu"],
            forward["phi"]
        )
        loss.backward()
        optimizer.step()
        losses.append({
            "loss": loss.item() * batch["sequence"].shape[0],
            "length": batch["sequence"].shape[0]
        })
        progress.set_postfix(loss=f"{loss.item():.3f}")

        if max_batches is not None and step >= max_batches:
            break
    return (
        sum(item["loss"] for item in losses)
        / sum(item["length"] for item in losses)
    )

def val_one_epoch(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    verbose = True
) -> float:
    model.eval()
    losses = []

    progress = tqdm(
        val_loader,
        desc="Validating epoch",
        unit=" batches",
        disable=not verbose
    )
    with torch.no_grad():
        for batch in progress:
            forward = model(batch["sequence"])
            loss = calculate_loss(
                batch["k"],
                batch["n"],
                forward["pi"],
                forward["mu"],
                forward["phi"]
            )
            losses.append({
                "loss": loss.item() * batch["sequence"].shape[0],
                "length": batch["sequence"].shape[0]
            })
            progress.set_postfix(loss=f"{loss.item():.3f}")

    return (
        sum(item["loss"] for item in losses)
        / sum(item["length"] for item in losses)
    )

def fit_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    max_epochs: int = 1000,
    checkpoint_path: str = "data/best_model.pt",
    patience: int = 5,
    min_delta: float = 1e-4,
    verbose: bool = True
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
            verbose=verbose
        )
        val_loss = val_one_epoch(model, val_loader, verbose=verbose)
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

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "concentration": phi,
                "overdisperson": psi
                }, checkpoint_path
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return history
