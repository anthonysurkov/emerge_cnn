import torch
from pathlib import Path
from typing import Callable

from .truth import set_seed
from .training import train_model
from .model import ConvModelFramework
from .paths import DATA_DIR


# Don't change me
SEQ_LENGTH  = 10
PHI_INIT    = 1
TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE   = 128

# Naming
MODEL_NAME = "twoconv_sharedhidden"


from .model import OneHotFeats, OneLayerConv, DenseHeads
BASELINE_NUM_FILTERS = 32
BASELINE_KERNEL_SIZE = 7
def assemble_baseline_model() -> torch.nn.Module:
    return ConvModelFramework(
        encoder_block=OneHotFeats(),
        conv_block=OneLayerConv(
            num_filters=BASELINE_NUM_FILTERS,
            kernel_size=BASELINE_KERNEL_SIZE
        ),
        heads_block=DenseHeads(
            BASELINE_NUM_FILTERS * (SEQ_LENGTH - BASELINE_KERNEL_SIZE + 1)),
        phi_init=PHI_INIT
    )

from .model import OneHotFeats, TwoLayerConv, DenseHeads
TWOCONV_NUM_FILTERS_LAYER_ONE = 32
TWOCONV_NUM_FILTERS_LAYER_TWO = 128
TWOCONV_KERNEL_SIZE_LAYER_ONE = 3
TWOCONV_KERNEL_SIZE_LAYER_TWO = 6
def assemble_twoconv_model() -> torch.nn.Module:
    return ConvModelFramework(
        encoder_block=OneHotFeats(),
        conv_block=TwoLayerConv(
            num_filters_layer_one=TWOCONV_NUM_FILTERS_LAYER_ONE,
            num_filters_layer_two=TWOCONV_NUM_FILTERS_LAYER_TWO,
            kernel_size_layer_one=TWOCONV_KERNEL_SIZE_LAYER_ONE,
            kernel_size_layer_two=TWOCONV_KERNEL_SIZE_LAYER_TWO
        ),
        heads_block=DenseHeads(
            TWOCONV_NUM_FILTERS_LAYER_TWO
            * (SEQ_LENGTH
               - TWOCONV_KERNEL_SIZE_LAYER_ONE
               - TWOCONV_KERNEL_SIZE_LAYER_TWO
               + 2
            )
        ),
        phi_init=PHI_INIT
    )

from .model import OneHotFeats, OneLayerConv, SharedHidden
SHAREDHIDDEN_NUM_FILTERS = 64
SHAREDHIDDEN_KERNEL_SIZE = 6
SHAREDHIDDEN_HIDDEN_SIZE = 32
def assemble_sharedhidden_model() -> torch.nn.Module:
    return ConvModelFramework(
        encoder_block=OneHotFeats(),
        conv_block=OneLayerConv(
            num_filters=SHAREDHIDDEN_NUM_FILTERS,
            kernel_size=SHAREDHIDDEN_KERNEL_SIZE
        ),
        heads_block=SharedHidden(
            input_size=(
                SHAREDHIDDEN_NUM_FILTERS
                * (SEQ_LENGTH - SHAREDHIDDEN_KERNEL_SIZE + 1)
            ),
            hidden_size=SHAREDHIDDEN_HIDDEN_SIZE
        ),
        phi_init=PHI_INIT
    )

from .model import OneHotFeats, OneLayerConv, TwoSharedHidden
TWOSHAREDHIDDEN_NUM_FILTERS = 64
TWOSHAREDHIDDEN_KERNEL_SIZE = 6
TWOSHAREDHIDDEN_HIDDEN_ONE = 32
TWOSHAREDHIDDEN_HIDDEN_TWO = 32
def assemble_twosharedhidden_model() -> torch.nn.Module:
    return ConvModelFramework(
        encoder_block=OneHotFeats(),
        conv_block=OneLayerConv(
            num_filters=TWOSHAREDHIDDEN_NUM_FILTERS,
            kernel_size=TWOSHAREDHIDDEN_KERNEL_SIZE
        ),
        heads_block=TwoSharedHidden(
            input_size=(
                TWOSHAREDHIDDEN_NUM_FILTERS
                * (SEQ_LENGTH - TWOSHAREDHIDDEN_KERNEL_SIZE + 1)
            ),
            hidden_size_one=TWOSHAREDHIDDEN_HIDDEN_ONE,
            hidden_size_two=TWOSHAREDHIDDEN_HIDDEN_TWO
        ),
        phi_init=PHI_INIT
    )

from .model import OneHotFeats, OneLayerConv, SplitHidden
SPLITHIDDEN_NUM_FILTERS = 64
SPLITHIDDEN_KERNEL_SIZE = 6
SPLITHIDDEN_HIDDEN_SIZE_PI = 32
SPLITHIDDEN_HIDDEN_SIZE_MU = 32
def assemble_splithidden_model() -> torch.nn.Module:
    return ConvModelFramework(
        encoder_block=OneHotFeats(),
        conv_block=OneLayerConv(
            num_filters=SPLITHIDDEN_NUM_FILTERS,
            kernel_size=SPLITHIDDEN_KERNEL_SIZE
        ),
        heads_block=SplitHidden(
            input_size=(
                SPLITHIDDEN_NUM_FILTERS
                * (SEQ_LENGTH - SPLITHIDDEN_KERNEL_SIZE + 1)
            ),
            hidden_size_pi=SPLITHIDDEN_HIDDEN_SIZE_PI,
            hidden_size_mu=SPLITHIDDEN_HIDDEN_SIZE_MU
        ),
        phi_init=PHI_INIT
    )

from .model import OneHotFeats, TwoLayerConv, SharedHidden
TWOC_SHAREDH_NUM_FILTERS_LAYER_ONE = 16
TWOC_SHAREDH_NUM_FILTERS_LAYER_TWO = 64
TWOC_SHAREDH_KERNEL_SIZE_LAYER_ONE = 3
TWOC_SHAREDH_KERNEL_SIZE_LAYER_TWO = 4
TWOC_SHAREDH_HIDDEN_SIZE = 32
def assemble_twoconv_sharedhidden_model() -> torch.nn.Module:
    return ConvModelFramework(
        encoder_block=OneHotFeats(),
        conv_block=TwoLayerConv(
            num_filters_layer_one=TWOC_SHAREDH_NUM_FILTERS_LAYER_ONE,
            num_filters_layer_two=TWOC_SHAREDH_NUM_FILTERS_LAYER_TWO,
            kernel_size_layer_one=TWOC_SHAREDH_KERNEL_SIZE_LAYER_ONE,
            kernel_size_layer_two=TWOC_SHAREDH_KERNEL_SIZE_LAYER_TWO
        ),
        heads_block=SharedHidden(
            input_size=(
                TWOC_SHAREDH_NUM_FILTERS_LAYER_TWO
                * (SEQ_LENGTH
                   - TWOC_SHAREDH_KERNEL_SIZE_LAYER_ONE
                   - TWOC_SHAREDH_KERNEL_SIZE_LAYER_TWO
                   + 2
                )
            ),
            hidden_size=TWOC_SHAREDH_HIDDEN_SIZE
        ),
        phi_init=PHI_INIT
    )


def main(model_assembler: Callable):
    seed=42
    set_seed(seed)

    if MODEL_NAME == "baseline":
        ckpt_path = (
            f"{DATA_DIR}/{MODEL_NAME}_model"
            f"_k{BASELINE_KERNEL_SIZE}"
            f"_f{BASELINE_NUM_FILTERS}"
            f"_ckpt.pt"
        )
    elif MODEL_NAME == "twoconv":
        ckpt_path = (
            f"{DATA_DIR}/{MODEL_NAME}_model"
            f"_k1{TWOCONV_KERNEL_SIZE_LAYER_ONE}"
            f"_k2{TWOCONV_KERNEL_SIZE_LAYER_TWO}"
            f"_f1{TWOCONV_NUM_FILTERS_LAYER_ONE}"
            f"_f2{TWOCONV_NUM_FILTERS_LAYER_TWO}"
            f"_ckpt.pt"
        )
    elif MODEL_NAME == "sharedhidden":
        ckpt_path = (
            f"{DATA_DIR}/{MODEL_NAME}_model"
            f"_k{SHAREDHIDDEN_KERNEL_SIZE}"
            f"_f{SHAREDHIDDEN_NUM_FILTERS}"
            f"_h{SHAREDHIDDEN_HIDDEN_SIZE}"
            f"_ckpt.pt"
        )
    elif MODEL_NAME == "splithidden":
        ckpt_path = (
            f"{DATA_DIR}/{MODEL_NAME}_model"
            f"_k{SPLITHIDDEN_KERNEL_SIZE}"
            f"_f{SPLITHIDDEN_NUM_FILTERS}"
            f"_hpi{SPLITHIDDEN_HIDDEN_SIZE_PI}"
            f"_hmu{SPLITHIDDEN_HIDDEN_SIZE_MU}"
            f"_ckpt.pt"
        )
    elif MODEL_NAME == "twoconv_sharedhidden":
        ckpt_path = (
            f"{DATA_DIR}/{MODEL_NAME}_model"
            f"_k1{TWOC_SHAREDH_KERNEL_SIZE_LAYER_ONE}"
            f"_k2{TWOC_SHAREDH_KERNEL_SIZE_LAYER_TWO}"
            f"_f1{TWOC_SHAREDH_NUM_FILTERS_LAYER_ONE}"
            f"_f2{TWOC_SHAREDH_NUM_FILTERS_LAYER_TWO}"
            f"_h{TWOC_SHAREDH_HIDDEN_SIZE}"
            f"_ckpt.pt"
        )
    elif MODEL_NAME == "twosharedhidden":
        ckpt_path = (
            f"{DATA_DIR}/{MODEL_NAME}_model"
            f"_k{TWOSHAREDHIDDEN_KERNEL_SIZE}"
            f"_f{TWOSHAREDHIDDEN_NUM_FILTERS}"
            f"_h1{TWOSHAREDHIDDEN_HIDDEN_ONE}"
            f"_h2{TWOSHAREDHIDDEN_HIDDEN_TWO}"
            f"_ckpt.pt"
        )

    else:
        raise ValueError(
            "Model name does not correspond to registered models"
        )

    train_model(
        model=model_assembler(),
        checkpoint_path=ckpt_path
    )


if __name__ == "__main__":
    model_assembler = globals()[f"assemble_{MODEL_NAME}_model"]
    main(model_assembler=model_assembler)
