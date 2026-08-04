from pathlib import Path
import torch

from truth import EmergeCNNPaths, load_train_val
from core import EmergeDataset
from training import fit_model
from encoders.one_hot import Features
from convs.one_layer import OneLayerConv
from heads.dense_heads import DenseHeads
from pipelines.one_conv_dense_model.model import OneConvDenseModel


NUM_FILTERS = 32
KERNEL_SIZE = 3
SEQ_LENGTH  = 10
PHI_INIT    = 1

TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE   = 128


def main():
    paths = EmergeCNNPaths(data_dir=Path("data"), screen_name="r255x")
    train_df, val_df = load_train_val(paths)

    train_data = EmergeDataset(df=train_df)
    val_data = EmergeDataset(df=val_df)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=VAL_BATCH_SIZE, shuffle=False
    )

    model = OneConvDenseModel(
        encoder_block=Features(),
        conv_block=OneLayerConv(
            num_filters=NUM_FILTERS, kernel_size=KERNEL_SIZE
        ),
        heads_block=DenseHeads(NUM_FILTERS * (SEQ_LENGTH - KERNEL_SIZE + 1)),
        phi_init = PHI_INIT
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    training_history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        max_epochs=3,
        patience=2
    )
    print(training_history)


if __name__ == "__main__":
    main()
