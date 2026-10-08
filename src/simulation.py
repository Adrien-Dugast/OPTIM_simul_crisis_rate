"""Simulation des trajectoires de prix : Black-Scholes, avec rupture de volatilité sur une partie des trajectoires."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Paths:
    """Un lot de M trajectoires sur N pas.

    S       : (M, N+1) prix aux dates de rebalancement, S[:, 0] = S0
    log_ret : (M, W+N) log-rendements, dont W jours avant t = 0 (pour la volatilité réalisée)
    sigma   : (M, N)   volatilité en vigueur sur le pas t -> t+1
    rupture : (M,)     date de passage en crise (N si la trajectoire reste calme)
    """
    S: torch.Tensor
    log_ret: torch.Tensor
    sigma: torch.Tensor
    rupture: torch.Tensor

    @property
    def crisis(self):
        """(M,) True pour les trajectoires qui passent en crise avant l'échéance."""
        return self.rupture < self.S.shape[1] - 1

    def __len__(self):
        return self.S.shape[0]


def gbm_paths(M, N, S0, r, sigma, dt, rng):
    """M trajectoires de Black-Scholes (mouvement brownien géométrique) sur N pas de temps dt.

    rng : numpy.random.Generator, pour que les lots soient reproductibles.
    Renvoie un tenseur float32 de forme (M, N+1), avec S[:, 0] = S0.
    """
    Z = rng.standard_normal((M, N))
    log_incr = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.cumsum(log_incr, axis=1)
    log_S = np.concatenate([np.zeros((M, 1)), log_S], axis=1)
    return torch.from_numpy(S0 * np.exp(log_S)).float()


def regime_paths(market, M, rng):
    """Black-Scholes à deux régimes (r = 0).

    Une proportion p_crisis des trajectoires passe en crise à une date tirée uniformément dans {0, ..., N-1} :
    la volatilité passe de sigma_calm à sigma_crisis jusqu'à l'échéance. Les vol_window jours qui précèdent
    t = 0 sont simulés en régime calme. Avec p_crisis = 0 et vol_window = 0, les tirages sont exactement ceux
    de gbm_paths (même graine, mêmes trajectoires).
    """
    N, W, dt = market.n_steps, market.vol_window, market.dt
    Z = rng.standard_normal((M, W + N))
    if market.p_crisis > 0:
        is_crisis = rng.random(M) < market.p_crisis
        rupture = np.where(is_crisis, rng.integers(0, N, M), N)
    else:
        rupture = np.full(M, N)
    sigma = np.where(np.arange(N)[None, :] >= rupture[:, None], market.sigma_crisis, market.sigma_calm)
    sigma_all = np.concatenate([np.full((M, W), market.sigma_calm), sigma], axis=1)

    log_ret = -0.5 * sigma_all**2 * dt + sigma_all * np.sqrt(dt) * Z
    log_S = np.concatenate([np.zeros((M, 1)), np.cumsum(log_ret[:, W:], axis=1)], axis=1)
    return Paths(S=torch.from_numpy(market.S0 * np.exp(log_S)).float(),
                 log_ret=torch.from_numpy(log_ret).float(),
                 sigma=torch.from_numpy(sigma).float(),
                 rupture=torch.from_numpy(rupture))


def realized_vol(paths, window, dt):
    """Volatilité réalisée annualisée à chaque date de décision t = 0, ..., N-1 : (M, N).
    sigma_hat_t = sqrt(mean des log-rendements^2 sur les window jours qui finissent en t / dt)."""
    N = paths.S.shape[1] - 1
    windows = paths.log_ret.unfold(1, window, 1)[:, :N]           # (M, N, window), fenêtre t-window+1, ..., t
    return torch.sqrt((windows**2).mean(dim=-1) / dt)
