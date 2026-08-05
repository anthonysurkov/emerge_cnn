import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Any, Callable
from dataclasses import dataclass
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

from .truth import EmergeDataset, EmergeCNNPaths, load_train_val
from .truth import SPLITS_SEED
from .model import ConvModelFramework
from .losses import betabinom_logprob


def load_model(
    checkpoint_path: str,
    model_assembly_func: Callable
) -> tuple[ConvModelFramework, Any]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )
    state_dict = checkpoint["model_state_dict"]

    model = model_assembly_func()
    model.load_state_dict(state_dict)
    model.to(device)

    return model, checkpoint

def append_model_preds(
    model: ConvModelFramework,
    val_df: pd.DataFrame,
    val_loader: torch.utils.data.DataLoader,
    splits_seed: int = SPLITS_SEED
) -> None:
    device = next(model.parameters()).device

    outputs = {"nonzero_prob": [], "mu": []}
    with torch.inference_mode():
        for batch in val_loader:
            sequence = batch["sequence"].to(device)
            n = batch["n"].to(device)
            pred = model(sequence)

            bb_zero_prob = betabinom_logprob(
                k=torch.zeros_like(n),
                n=n,
                mu=pred["mu"],
                phi=pred["phi"]
            ).exp()
            nonzero_prob = (1.0 - pred["pi"]) * (1.0 - bb_zero_prob)

            outputs["nonzero_prob"].append(nonzero_prob.cpu())
            outputs["mu"].append(pred["mu"].cpu())

    editing_prob = torch.cat(outputs["nonzero_prob"]).numpy()
    continuous_editing = torch.cat(outputs["mu"]).numpy()

    val_df.reset_index(drop=True, inplace=True)
    val_df["pred_classifier_score"] = editing_prob
    val_df["pred_classifier"] = editing_prob > 0.5
    val_df["pred_regressor"] = continuous_editing

def _get_val(
    path: EmergeCNNPaths,
    splits_seed: int = SPLITS_SEED
) -> tuple[pd.DataFrame, torch.utils.data.DataLoader]:
    _, val_df = load_train_val(path, seed=splits_seed)
    val_loader = DataLoader(
        EmergeDataset(val_df),
        batch_size=4096,
        shuffle=False
    )
    return val_df, val_loader

def _append_editing_class(val_df: pd.DataFrame) -> None:
    """Add the observed event modeled by the ZIBB mixture classifier."""
    val_df["editing_status"] = val_df["k"] > 0

def eval_model(
    model: ConvModelFramework,
    model_config: dict[str, Any],
    screen_name: str = "r255x",
    splits_seed: int = SPLITS_SEED
):
    device = next(model.parameters()).device
    model.eval()

    path = EmergeCNNPaths(screen_name=screen_name)
    val_df, val_loader = _get_val(path=path, splits_seed=splits_seed)

    append_model_preds(
        model=model,
        val_df=val_df,
        val_loader=val_loader,
        splits_seed=splits_seed
    )
    _append_editing_class(val_df)

    editing_rows = val_df["editing_status"]
    regress_y_true = val_df.loc[editing_rows, "mle"]
    regress_y_pred = val_df.loc[editing_rows, "pred_regressor"]
    class_y_true = val_df["editing_status"]
    class_y_score = val_df["pred_classifier_score"]
    class_y_pred = val_df["pred_classifier"]

    return ({
        "model_config": model_config,
        "r2": r2_score(regress_y_true, regress_y_pred),
        "mae": mean_absolute_error(regress_y_true, regress_y_pred),
        "rmse": root_mean_squared_error(regress_y_true, regress_y_pred),
        "classification_target": "k > 0",
        "threshold": 0,
        "classifier_score_threshold": 0.5,
        "auroc": roc_auc_score(class_y_true, class_y_score),
        "auprc": average_precision_score(class_y_true, class_y_score),
        "accuracy": accuracy_score(class_y_true, class_y_pred),
        "f1": f1_score(class_y_true, class_y_pred, zero_division=0),
        "precision": precision_score(
            class_y_true,
            class_y_pred,
            zero_division=0
        ),
        "recall": recall_score(class_y_true, class_y_pred, zero_division=0)
    })


from .model import OneHotFeats, OneLayerConv, DenseHeads
from .main import assemble_baseline_model

def eval_main(checkpoint_path: str):
    checkpoint_Path = Path(checkpoint_path)
    assert checkpoint_Path.is_file()

    model, checkpoint = load_model(
        checkpoint_path=checkpoint_path,
        model_assembly_func=assemble_baseline_model
    )
    statistics = eval_model(
        model=model,
        model_config=checkpoint["metadata"]["model_config"],
        screen_name="r255x"
    )
    print(statistics)


if __name__ == "__main__":
    eval_main(checkpoint_path="data/baseline_model_k7_f32_ckpt.pt")
