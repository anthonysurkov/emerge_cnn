import torch
from pathlib import Path
from typing import Callable

from .truth import set_seed
from .training import train_model


def main():
    set_seed()

    train_model(
        model=model_assembler(),
        checkpoint_path=ckpt_path
    )


if __name__ == "__main__":
    main()
