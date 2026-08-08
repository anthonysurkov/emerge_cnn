import pandas as pd
import numpy as np
import torch
import random
from dataclasses import dataclass, field
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from .paths import DATA_DIR


NO_EDITING_CUTOFF = 0.05
SEED = 42
SPLITS_SEED = 42
NT_MAP = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}

EDITING_STRATUM_UPPER_BOUNDS = (0.0, 0.02, 0.10, 0.40, 0.64)


class EmergeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        loss_weights: pd.Series | np.ndarray | None = None
    ):
        self.df = df.reset_index(drop=True)
        self.loss_weights = None
        if loss_weights is not None:
            weights = np.asarray(loss_weights, dtype=np.float32)
            if weights.ndim != 1 or len(weights) != len(self.df):
                raise ValueError(
                    "loss_weights must be one-dimensional and match df length"
                )
            if not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError(
                    "loss_weights must contain finite, nonnegative values"
                )
            self.loss_weights = weights

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        sequence = torch.tensor(
            [NT_MAP[char] for char in row["5to3"]],
            dtype=torch.long
        )
        n = torch.tensor(row["n"], dtype=torch.float32)
        k = torch.tensor(row["k"], dtype=torch.float32)
        mle = torch.tensor(row["mle"], dtype=torch.float32)

        item = {"sequence": sequence, "n": n, "k": k, "mle": mle}
        if self.loss_weights is not None:
            item["loss_weight"] = torch.tensor(
                self.loss_weights[index],
                dtype=torch.float32
            )
        return item


@dataclass
class EmergeCNNPaths:
    screen_name: str

    data_dir: Path = DATA_DIR
    screen_path: Path = field(init=False)
    train_path: Path = field(init=False)
    val_path: Path = field(init=False)
    test_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)

        self.screen_path = self.data_dir / f"{self.screen_name}.csv"
        self.train_path = self.data_dir / f"{self.screen_name}_train_idx.csv"
        self.val_path = self.data_dir / f"{self.screen_name}_val_idx.csv"
        self.test_path = self.data_dir / f"{self.screen_name}_test_idx.csv"

        if not self.screen_path.is_file():
            raise FileNotFoundError(
                f"Screen file does not exist: {self.screen_path}"
            )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def splits_exist(paths: EmergeCNNPaths) -> bool:
    return (
        paths.train_path.is_file()
        and paths.test_path.is_file()
        and paths.val_path.is_file()
    )

def make_splits(
    df: pd.DataFrame,
    paths: EmergeCNNPaths,
    splits_seed: int = SPLITS_SEED,
    force_regenerate: bool = False
) -> None:
    if splits_exist(paths) and not force_regenerate:
        return

    def split_df(df: pd.DataFrame) -> tuple[pd.Index, pd.Index, pd.Index]:
        train, remain = train_test_split(
            df,
            test_size=0.2,
            stratify=editing_strata(df["mle"]),
            random_state=splits_seed
        )
        test, val = train_test_split(
            remain,
            test_size=0.5,
            stratify=editing_strata(remain["mle"]),
            random_state=splits_seed
        )
        return train.index, val.index, test.index

    pos = split_df(df[df["mle"] > NO_EDITING_CUTOFF])
    zer = split_df(df[df["mle"] <= NO_EDITING_CUTOFF])
    train_idx = pos[0].append(zer[0])
    val_idx = pos[1].append(zer[1])
    test_idx = pos[2].append(zer[2])

    rng = np.random.default_rng(splits_seed)
    train_idx = pd.Index(rng.permutation(train_idx))
    val_idx = pd.Index(rng.permutation(val_idx))
    test_idx = pd.Index(rng.permutation(test_idx))

    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)

    pd.Series(train_idx, name="idx").to_csv(paths.train_path, index=False)
    pd.Series(val_idx, name="idx").to_csv(paths.val_path, index=False)
    pd.Series(test_idx, name="idx").to_csv(paths.test_path, index=False)

def load_train_val(
    paths: EmergeCNNPaths,
    seed: int = 42,
    force_regenerate: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(paths.screen_path, usecols=["5to3", "n", "k", "mle"])
    if not splits_exist(paths) or force_regenerate:
        make_splits(df, paths, splits_seed=seed, force_regenerate=force_regenerate)

    train_idx = pd.read_csv(paths.train_path)["idx"]
    train_df = df.loc[train_idx]
    val_idx = pd.read_csv(paths.val_path)["idx"]
    val_df = df.loc[val_idx]
    return train_df, val_df

def load_test(
    paths: EmergeCNNPaths,
    seed: int = 42,
    force_regenerate: bool = False
) -> pd.DataFrame:
    df = pd.read_csv(paths.screen_path, usecols=["5to3", "n", "k", "mle"])
    if not splits_exist(paths) or force_regenerate:
        make_splits(df, paths, seed=seed, force_regenerate=force_regenerate)

    test_idx = pd.read_csv(paths.test_path)["idx"]
    test_df = df.loc[test_idx]
    return test_df

def editing_strata(mle: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(mle, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("mle must be one-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("mle must contain only finite values")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("mle values must be between 0 and 1")

    return np.searchsorted(
        EDITING_STRATUM_UPPER_BOUNDS,
        values,
        side="left"
    )

def stratified_kfolds(n_splits: int, splits_seed: int = SPLITS_SEED):
    strata = editing_strata(df["mle"])
    kfold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=splits_seed
    )
    k_train_vals: list[tuple[EmergeDataset, EmergeDataset]]
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df, strata)):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        train_dataset = EmergeDataset(train_df)
        val_dataset = EmergeDataset(val_df)
        k_train_vals.append(train_dataset, val_dataset)
    return k_train_vals
