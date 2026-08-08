import inspect
from typing import Any

import torch

from . import model


def _resolve_module_class(name: str) -> type[torch.nn.Module]:
    cls = getattr(model, name, None)

    if not isinstance(cls, type) or not issubclass(cls, torch.nn.Module):
        raise TypeError(f"{name!r} is not a torch module class")

    return cls

def _get_model_blocks(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint["metadata"]["model_config"]
    state_dict = checkpoint["model_state_dict"]

    encoder_class = _resolve_module_class(config["encoder_class"])
    conv_class = _resolve_module_class(config["conv_class"])
    heads_class = _resolve_module_class(config["heads_class"])

    if config["conv_class"] == "OneLayerConv":
        conv_weight = state_dict["conv_block.layer.weight"]
        conv_kwargs = {
            "num_filters": conv_weight.shape[0],
            "kernel_size": conv_weight.shape[2],
        }
    elif config["conv_class"] == "TwoLayerConv":
        layer_one = state_dict["conv_block.layer_one.weight"]
        layer_two = state_dict["conv_block.layer_two.weight"]
        conv_kwargs = {
            "num_filters_layer_one": layer_one.shape[0],
            "num_filters_layer_two": layer_two.shape[0],
            "kernel_size_layer_one": layer_one.shape[2],
            "kernel_size_layer_two": layer_two.shape[2],
        }
    else:
        raise TypeError(
            f"Unsupported convolution block {config['conv_class']!r}"
        )

    heads_signature = inspect.signature(heads_class)
    heads_kwargs = {
        name: value
        for name, value in config["heads"].items()
        if name in heads_signature.parameters
    }

    return {
        "encoder_block": encoder_class(),
        "conv_block": conv_class(**conv_kwargs),
        "heads_block": heads_class(**heads_kwargs),
        "phi_init": state_dict["phi_raw"].item(),
    }

def model_from_checkpoint(
    checkpoint: dict[str, Any]
) -> model.ConvModelFramework:
    config = checkpoint["metadata"]["model_config"]
    model_class = _resolve_module_class(config["model_class"])

    if not issubclass(model_class, model.ConvModelFramework):
        raise TypeError(
            f"Unsupported model framework {config['model_class']!r}"
        )

    cnn = model_class(**_get_model_blocks(checkpoint))
    cnn.load_state_dict(checkpoint["model_state_dict"])
    return cnn

def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
