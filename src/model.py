from dataclasses import dataclass, asdict
from itertools import product
from typing import Any

from .blocks import (
    ConvModelFramework,
    ConvStack,
    DenseHeads,
    OneHotFeats,
    SharedHidden,
    SplitHidden
)


def serialize_head(spec) -> dict:
    if isinstance(spec, DirectHeadSpec):
        return {"type": "direct"}
    if isinstance(spec, SharedHeadSpec):
        return {"type": "shared", "hidden_size": spec.hidden_size}
    if isinstance(spec, SplitHeadSpec):
        return {
            "type": "split",
            "pi_hidden_size": spec.pi_hidden_size,
            "mu_hidden_size": spec.mu_hidden_size
        }

@dataclass(frozen=True)
class ConvLayerSpec:
    filters: int
    kernel_size: int

@dataclass(frozen=True)
class DirectHeadSpec:
    pass

@dataclass(frozen=True)
class SharedHeadSpec:
    hidden_size: int

@dataclass(frozen=True)
class SplitHeadSpec:
    pi_hidden_size: int
    mu_hidden_size: int

HeadSpec = DirectHeadSpec | SharedHeadSpec | SplitHeadSpec


@dataclass(frozen=True)
class ModelSpec:
    preset_id: str
    conv_layers: tuple[ConvLayerSpec, ...]
    heads: HeadSpec
    sequence_length: int = 10
    phi_init: float = 1.0

    def flattened_size(self) -> int:
        output_length = self.sequence_length
        for layer in self.conv_layers:
            output_length -= layer.kernel_size - 1
        if output_length < 1:
            raise ValueError(
                f"{self.preset_id!r} reduces seq below one pos"
            )
        return self.conv_layers[-1].filters * output_length

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_model(spec: ModelSpec) -> ConvModelFramework:
    conv = ConvStack(
        in_channels=4,
        layers=tuple(
            (layer.filters, layer.kernel_size)
            for layer in spec.conv_layers
        ),
    )
    input_size = spec.flattened_size()

    if isinstance(spec.heads, DirectHeadSpec):
        heads = DenseHeads(input_size=input_size)
    elif isinstance(spec.heads, SharedHeadSpec):
        heads = SharedHidden(
            input_size=input_size,
            hidden_size=spec.heads.hidden_size
        )
    elif isinstance(spec.heads, SplitHeadSpec):
        heads = SplitHidden(
            input_size=input_size,
            hidden_size_pi=spec.heads.pi_hidden_size,
            hidden_size_mu=spec.heads.mu_hidden_size
        )
    else:
        raise TypeError(f"Unsupported head: {spec.heads!r}")

    return ConvModelFramework(
        encoder_block=OneHotFeats(),
        conv_block=conv,
        heads_block=heads,
        phi_init=spec.phi_init
    )


# PRESETS

def config_conv2_sharedmlp(
    f1: int,
    k1: int,
    f2: int,
    k2: int,
    h: int
) -> ModelSpec:
    return ModelSpec(
        preset_id = f"conv2-f1-{f1}-k1-{k1}-f2-{f2}-k2-{k2}-sharedmlp-{h}",
        conv_layers=(
            ConvLayerSpec(filters=f1, kernel_size=k1),
            ConvLayerSpec(filters=f2, kernel_size=k2)
        ),
        heads=SharedHeadSpec(hidden_size=h)
    )

def scanconfig_conv2_sharedmlp(
    f1_range: list[int],
    k1_range: list[int],
    f2_range: list[int],
    k2_range: list[int],
    h_range: list[int]
) -> list[ModelSpec]:
    return [
        config_conv2_sharedmlp(f1, k1, f2, k2, h)
        for f1, k1, f2, k2, h
        in product(f1_range, k1_range, f2_range, k2_range, h_range)
    ]
