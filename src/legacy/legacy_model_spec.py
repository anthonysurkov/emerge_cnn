# DEPRECATED INCL. IMPORTS


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

