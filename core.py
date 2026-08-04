import pandas as pd
import torch


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


def log_choose(n: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.lgamma(n+1) - torch.lgamma(k+1) - torch.lgamma(n-k+1)

def log_beta(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha+beta)

def betabinom_logprob(
    k: torch.Tensor,
    n: torch.Tensor,
    mu: torch.Tensor,
    phi: torch.Tensor
) -> torch.Tensor:
    mu = mu.clamp(min=1e-6, max=1-1e-6)
    phi = phi.clamp_min(1e-6)
    alpha = mu * phi
    beta = (1 - mu) * phi
    return (
        log_choose(n, k)
        + log_beta(k+alpha, n-k+beta)
        - log_beta(alpha, beta)
    )

def calculate_loss(
    k: torch.Tensor,
    n: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    phi: torch.Tensor
) -> torch.Tensor:
    pi = pi.clamp(min=1e-6, max=1-1e-6)

    bb_logprob = betabinom_logprob(k, n, mu, phi)
    log_bb_component = torch.log1p(-pi) + bb_logprob
    zero_logprob = torch.logaddexp(
        torch.log(pi),
        log_bb_component
    )

    observation_logprob = torch.where(
        k == 0,
        zero_logprob,
        log_bb_component
    )
    return -observation_logprob.mean()
