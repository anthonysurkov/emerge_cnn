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
    phi: torch.Tensor,
    sample_weight: torch.Tensor | None = None
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
    per_example_loss = -observation_logprob

    if sample_weight is None:
        return per_example_loss.mean()

    if sample_weight.shape != per_example_loss.shape:
        raise ValueError("sample_weight must match the per-example loss shape")
    if not torch.isfinite(sample_weight).all().item():
        raise ValueError("sample_weight must contain only finite values")
    if (sample_weight < 0).any().item():
        raise ValueError("sample_weight must be nonnegative")

    weight_sum = sample_weight.sum()
    if weight_sum.item() <= 0:
        raise ValueError("sample_weight must have a positive sum")
    return (sample_weight * per_example_loss).sum() / weight_sum
