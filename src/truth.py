import pandas as pd
import numpy as np
import torch
import random
from dataclasses import dataclass, field
from pathlib import Path
from sklearn.model_selection import train_test_split

from .paths import DATA_DIR


NO_EDITING_CUTOFF = 0.05
NT_MAP = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}


class EmergeDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

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

        return {"sequence": sequence, "n": n, "k": k, "mle": mle}


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
    seed: int = 42,
    force_regenerate: bool = False
) -> None:
    if splits_exist(paths) and not force_regenerate:
        return

    def split_df(df: pd.DataFrame) -> tuple[pd.Index, pd.Index, pd.Index]:
        train, remain = train_test_split(df, test_size=0.2, random_state=seed)
        test, val = train_test_split(remain, test_size=0.5, random_state=seed)
        return train.index, val.index, test.index

    pos = split_df(df[df["mle"] > NO_EDITING_CUTOFF])
    zer = split_df(df[df["mle"] <= NO_EDITING_CUTOFF])
    train_idx = pos[0].append(zer[0])
    val_idx = pos[1].append(zer[1])
    test_idx = pos[2].append(zer[2])

    rng = np.random.default_rng(seed)
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
        make_splits(df, paths, seed=seed, force_regenerate=force_regenerate)

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

