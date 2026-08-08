import unittest

import numpy as np
import torch
from scipy.stats import betabinom

from src.losses import betabinom_logprob, calculate_betabinom_loss


class BetaBinomialLossTests(unittest.TestCase):
    def test_logprob_matches_scipy(self):
        k = torch.tensor([0.0, 1.0, 4.0, 9.0], dtype=torch.float64)
        n = torch.tensor([10.0, 10.0, 12.0, 15.0], dtype=torch.float64)
        mu = torch.tensor([0.05, 0.20, 0.45, 0.70], dtype=torch.float64)
        phi = torch.tensor(3.5, dtype=torch.float64)

        actual = betabinom_logprob(k, n, mu, phi).detach().numpy()
        alpha = mu.numpy() * phi.item()
        beta = (1.0 - mu.numpy()) * phi.item()
        expected = betabinom.logpmf(k.numpy(), n.numpy(), alpha, beta)

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_zero_inflated_loss_matches_direct_probability(self):
        k = torch.tensor([0.0, 0.0, 2.0, 7.0], dtype=torch.float64)
        n = torch.tensor([10.0, 25.0, 10.0, 12.0], dtype=torch.float64)
        pi = torch.tensor([0.75, 0.20, 0.40, 0.05], dtype=torch.float64)
        mu = torch.tensor([0.10, 0.35, 0.25, 0.60], dtype=torch.float64)
        phi = torch.tensor(4.0, dtype=torch.float64)

        actual = calculate_betabinom_loss(k, n, pi, mu, phi).item()

        alpha = mu.numpy() * phi.item()
        beta = (1.0 - mu.numpy()) * phi.item()
        bb_probability = betabinom.pmf(k.numpy(), n.numpy(), alpha, beta)
        probability = np.where(
            k.numpy() == 0,
            pi.numpy() + (1.0 - pi.numpy()) * bb_probability,
            (1.0 - pi.numpy()) * bb_probability,
        )
        expected = -np.log(probability).mean()

        self.assertAlmostEqual(actual, expected, places=10)

    def test_one_head_loss_is_plain_beta_binomial(self):
        k = torch.tensor([0.0, 2.0, 7.0], dtype=torch.float64)
        n = torch.tensor([10.0, 10.0, 12.0], dtype=torch.float64)
        mu = torch.tensor([0.10, 0.25, 0.60], dtype=torch.float64)
        phi = torch.tensor(4.0, dtype=torch.float64)

        actual = calculate_betabinom_loss(k, n, None, mu, phi)
        expected = -betabinom_logprob(k, n, mu, phi).mean()

        torch.testing.assert_close(actual, expected)

    def test_loss_backpropagates_to_all_parameters(self):
        k = torch.tensor([0.0, 1.0, 3.0])
        n = torch.tensor([10.0, 10.0, 12.0])
        pi = torch.tensor([0.70, 0.20, 0.10], requires_grad=True)
        mu = torch.tensor([0.10, 0.25, 0.50], requires_grad=True)
        phi = torch.tensor(2.0, requires_grad=True)

        loss = calculate_betabinom_loss(k, n, pi, mu, phi)
        loss.backward()

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(loss))
        for parameter in (pi, mu, phi):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_weighted_loss_is_weighted_mean_of_per_example_losses(self):
        k = torch.tensor([0.0, 2.0, 7.0], dtype=torch.float64)
        n = torch.tensor([10.0, 10.0, 12.0], dtype=torch.float64)
        pi = torch.tensor([0.75, 0.40, 0.05], dtype=torch.float64)
        mu = torch.tensor([0.10, 0.25, 0.60], dtype=torch.float64)
        phi = torch.tensor(4.0, dtype=torch.float64)
        weights = torch.tensor([1.0, 2.0, 8.0], dtype=torch.float64)

        actual = calculate_betabinom_loss(
            k, n, pi, mu, phi, sample_weight=weights
        )
        individual_losses = torch.stack([
            calculate_betabinom_loss(
                k[i:i + 1],
                n[i:i + 1],
                pi[i:i + 1],
                mu[i:i + 1],
                phi,
            )
            for i in range(len(k))
        ])
        expected = (individual_losses * weights).sum() / weights.sum()

        torch.testing.assert_close(actual, expected)

    def test_weighted_loss_rejects_zero_total_weight(self):
        values = torch.tensor([0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "positive sum"):
            calculate_betabinom_loss(
                k=values,
                n=torch.ones_like(values),
                pi=torch.full_like(values, 0.5),
                mu=torch.full_like(values, 0.5),
                phi=torch.tensor(1.0),
                sample_weight=torch.zeros_like(values),
            )


if __name__ == "__main__":
    unittest.main()
