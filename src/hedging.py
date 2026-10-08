"""Réseau de couverture, entrées, P&L, pertes et références de Black-Scholes (call européen, r = 0)."""

import math

import torch

from src.simulation import realized_vol


# --- Références de Black-Scholes ---

def norm_cdf(x):
    """Fonction de répartition de la loi normale centrée réduite (tenseur)."""
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def bs_call_price(S0, K, sigma, T):
    """Prix de Black-Scholes d'un call avec r = 0 (scalaires, T en années)."""
    d1 = (math.log(S0 / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return S0 * Phi(d1) - K * Phi(d2)


def bs_delta(S, K, sigma, tau):
    """Delta BS N(d1) avec r = 0. S : prix (tenseur), sigma et tau (années, > 0) : scalaires ou tenseurs."""
    sigma = torch.as_tensor(sigma, dtype=S.dtype)
    tau = torch.as_tensor(tau, dtype=S.dtype)
    d1 = (torch.log(S / K) + 0.5 * sigma**2 * tau) / (sigma * torch.sqrt(tau))
    return norm_cdf(d1)


def time_to_maturity(market):
    """Temps restant en années aux dates de décision t = 0, ..., N-1 : (N,)."""
    N = market.n_steps
    return (N - torch.arange(N, dtype=torch.float32)) * market.dt


def oracle_delta(paths, market):
    """N(d1) avec la volatilité du régime en vigueur : couverture qui connaît le régime dès la rupture. (M, N)"""
    return bs_delta(paths.S[:, :-1], market.K, paths.sigma, time_to_maturity(market))


def realized_vol_delta(paths, market):
    """N(d1) avec la volatilité réalisée : référence sans oracle, qui ne voit la crise qu'avec retard. (M, N)"""
    sigma_hat = realized_vol(paths, market.vol_window, market.dt)
    return bs_delta(paths.S[:, :-1], market.K, sigma_hat, time_to_maturity(market))


# --- Entrées du réseau ---

class Features:
    """Entrées normalisées, d'ordre 1, aux dates de décision t = 0, ..., N-1 :
     - "tau"          : temps restant / T, soit (N - t)/N
     - "moneyness"    : ln(S_t / K)
     - "realized_vol" : ln(sigma_hat_t), centré-réduit avec la moyenne et l'écart-type de calibration.
    """

    def __init__(self, names, market, log_vol_mean=0.0, log_vol_std=1.0):
        self.names, self.market = list(names), market
        self.log_vol_mean, self.log_vol_std = log_vol_mean, log_vol_std

    @classmethod
    def calibrate(cls, names, market, calib_paths):
        """Fixe la normalisation de la volatilité réalisée sur un échantillon de calibration."""
        if "realized_vol" not in names:
            return cls(names, market)
        log_vol = torch.log(realized_vol(calib_paths, market.vol_window, market.dt))
        return cls(names, market, log_vol.mean().item(), log_vol.std().item())

    def state(self):
        return {"log_vol_mean": self.log_vol_mean, "log_vol_std": self.log_vol_std}

    def __len__(self):
        return len(self.names)

    def _stack(self, tau, S, sigma_hat):
        cols = {"tau": lambda: tau,
                "moneyness": lambda: torch.log(S / self.market.K),
                "realized_vol": lambda: (torch.log(sigma_hat) - self.log_vol_mean) / self.log_vol_std}
        return torch.stack([cols[name]() for name in self.names], dim=-1)

    def __call__(self, paths):
        """Paths -> (M, N, n_features)."""
        N = self.market.n_steps
        S = paths.S[:, :-1]
        tau = ((N - torch.arange(N, dtype=S.dtype)) / N).expand_as(S)
        sigma_hat = realized_vol(paths, self.market.vol_window, self.market.dt) if "realized_vol" in self.names else None
        return self._stack(tau, S, sigma_hat)

    def on_grid(self, S_grid, t, sigma_hat=None):
        """Entrées pour une grille de prix à la date t, avec une volatilité réalisée donnée : (len(S_grid), n_features)."""
        N = self.market.n_steps
        sigma_hat = None if sigma_hat is None else torch.full_like(S_grid, sigma_hat)
        return self._stack(torch.full_like(S_grid, (N - t) / N), S_grid, sigma_hat)


# --- Réseau de couverture ---

class HedgeNet(torch.nn.Module):
    """MLP appliqué à chaque date : x (M, N, n_inputs) -> delta (M, N)."""

    def __init__(self, n_inputs=2, width=64):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, width), torch.nn.ReLU(),
            torch.nn.Linear(width, width), torch.nn.ReLU(),
            torch.nn.Linear(width, 1),          # sortie linéaire : delta_t non borné
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


# --- P&L et pertes ---

def call_pnl(S, delta, p0, K):
    """P&L du vendeur du call couvert : p0 + sum_t delta_t (S_{t+1} - S_t) - (S_T - K)^+.
    S : (M, N+1), delta : (M, N) -> (M,)."""
    payoff = torch.clamp(S[:, -1] - K, min=0)
    gains = (delta * torch.diff(S, dim=1)).sum(dim=1)
    return p0 + gains - payoff


def mse_loss(pnl):
    """E[P&L^2] : on ne veut ni perte ni gain, seulement une couverture la plus exacte possible."""
    return (pnl ** 2).mean()


LOSSES = {"mse": mse_loss}
