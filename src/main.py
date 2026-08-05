from pathlib import Path
import torch

from .truth import set_seed
from .training import train_model
from .model import ConvModelFramework


NUM_FILTERS = 32
KERNEL_SIZE = 3
SEQ_LENGTH  = 10
PHI_INIT    = 1

TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE   = 128


from .model import OneHotFeats, OneLayerConv, DenseHeads
def assemble_baseline_model() -> torch.nn.Module:
    return ConvModelFramework(
        encoder_block=OneHotFeats(),
        conv_block=OneLayerConv(
            num_filters=NUM_FILTERS, kernel_size=KERNEL_SIZE
        ),
        heads_block=DenseHeads(NUM_FILTERS * (SEQ_LENGTH - KERNEL_SIZE + 1)),
        phi_init=PHI_INIT
    )


def main():
    seed=42
    set_seed(seed)

    training_history = train_model(
        model=assemble_baseline_model()
    )


if __name__ == "__main__":
    main()
