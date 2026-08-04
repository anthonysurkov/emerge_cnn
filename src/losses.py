import torch


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

def calculate_betabinom_loss(
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
