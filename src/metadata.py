from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .blocks import ConvStack
from .truth import (
    EDITING_STRATUM_UPPER_BOUNDS,
    EmergeCNNPaths,
    NO_EDITING_CUTOFF,
    SPLITS_SEED,
)


def _as_list(value: int | tuple[int, ...]) -> list[int]:
    if isinstance(value, tuple):
        return list(value)
    return [value]

def get_model_config(model: torch.nn.Module) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_class": type(model).__name__,
        "encoder_class": type(model.encoder_block).__name__,
        "conv_class": type(model.conv_block).__name__,
        "heads_class": type(model.heads_block).__name__,
        "parameter_count": (
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "phi_raw_initial": float(model.phi_raw.detach().cpu().item()),
    }

    conv = getattr(model.conv_block, "layer", None)
    if isinstance(conv, torch.nn.Conv1d):
        config["convolution"] = {
            "in_channels": conv.in_channels,
            "out_channels": conv.out_channels,
            "kernel_size": _as_list(conv.kernel_size),
            "stride": _as_list(conv.stride),
            "padding": _as_list(conv.padding),
            "dilation": _as_list(conv.dilation),
            "groups": conv.groups,
            "bias": conv.bias is not None,
        }
    elif isinstance(model.conv_block, ConvStack):
        convolutions = model.conv_block.convolutions
        config["convolution"] = {
            "in_channels": convolutions[0].in_channels,
            "layers": [
                [layer.out_channels, layer.kernel_size[0]]
                for layer in convolutions
            ],
        }

    heads_config = getattr(model.heads_block, "config", None)
    if heads_config is not None:
        config["heads"] = dict(heads_config)

    return config

def get_training_config(
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    tail_weight_power: float,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "optimizer_class": type(optimizer).__name__,
        "learning_rates": [group["lr"] for group in optimizer.param_groups],
        "train_batch_size": train_loader.batch_size,
        "val_batch_size": val_loader.batch_size,
        "train_examples": len(train_loader.dataset),
        "val_examples": len(val_loader.dataset),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "tail_weight_power": tail_weight_power,
        "editing_stratum_upper_bounds": list(EDITING_STRATUM_UPPER_BOUNDS),
    }

def get_data_config(
    paths: EmergeCNNPaths,
    *,
    split_seed: int = SPLITS_SEED,
    no_editing_cutoff: float = NO_EDITING_CUTOFF,
) -> dict[str, Any]:
    def path_string(path: Path) -> str:
        return str(path.resolve())

    return {
        "screen_name": paths.screen_name,
        "screen_path": path_string(paths.screen_path),
        "train_split_path": path_string(paths.train_path),
        "val_split_path": path_string(paths.val_path),
        "test_split_path": path_string(paths.test_path),
        "split_seed": split_seed,
        "no_editing_cutoff": no_editing_cutoff,
    }
