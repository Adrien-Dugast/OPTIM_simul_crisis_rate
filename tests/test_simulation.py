import numpy as np
import torch

from src.config import MarketConfig
from src.simulation import gbm_paths, realized_vol, regime_paths


def test_no_crisis_reproduces_gbm():
    """Sans crise ni pré-historique, mêmes tirages que gbm_paths : E0 reste reproductible."""
    market = MarketConfig(sigma_calm=0.2, p_crisis=0.0, vol_window=0)
    paths = regime_paths(market, 100, np.random.default_rng(0))
    S = gbm_paths(100, market.n_steps, market.S0, 0.0, 0.2, market.dt, np.random.default_rng(0))
    assert torch.equal(paths.S, S)
    assert not paths.crisis.any()


def test_crisis_proportion_and_vol_switch():
    market = MarketConfig(sigma_calm=0.1, sigma_crisis=0.5, p_crisis=0.1, vol_window=10)
    paths = regime_paths(market, 20000, np.random.default_rng(0))
    assert abs(paths.crisis.float().mean().item() - 0.1) < 0.01
    N = market.n_steps
    t = torch.arange(N)[None, :]
    in_crisis = t >= paths.rupture[:, None]
    assert torch.all(paths.sigma[in_crisis] == 0.5) and torch.all(paths.sigma[~in_crisis] == 0.1)
    assert paths.log_ret.shape == (20000, 10 + N)
    assert torch.allclose(paths.S[:, 1:], market.S0 * torch.exp(paths.log_ret[:, 10:].double().cumsum(1)).float())


def test_realized_vol_matches_regime():
    market = MarketConfig(sigma_calm=0.1, sigma_crisis=0.5, p_crisis=0.5, vol_window=10)
    paths = regime_paths(market, 20000, np.random.default_rng(0))
    rv = realized_vol(paths, market.vol_window, market.dt)
    assert rv.shape == (20000, market.n_steps)
    t = torch.arange(market.n_steps)[None, :]
    calm = t < paths.rupture[:, None]
    informed = t >= paths.rupture[:, None] + market.vol_window
    assert abs(rv[calm].mean().item() - 0.1) < 0.01
    assert abs(rv[informed].mean().item() - 0.5) < 0.05
    assert abs(rv[:, 0].mean().item() - 0.1) < 0.01            # pré-historique calme : définie dès t = 0
