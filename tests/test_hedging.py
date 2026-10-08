import math

import numpy as np
import torch

from src.config import MarketConfig
from src.hedging import Features, bs_call_price, bs_delta, call_pnl, mse_loss, oracle_delta
from src.simulation import gbm_paths, regime_paths


def test_pnl_by_hand():
    S = torch.tensor([[100.0, 102.0, 105.0]])
    delta = torch.tensor([[0.5, 0.8]])
    # p0 + 0.5 * 2 + 0.8 * 3 - (105 - 100)
    assert torch.allclose(call_pnl(S, delta, p0=4.0, K=100.0), torch.tensor([4.0 + 1.0 + 2.4 - 5.0]))


def test_features_shape_and_values():
    market = MarketConfig(n_steps=4, vol_window=0)
    paths = regime_paths(market, 3, np.random.default_rng(0))
    paths.S[:] = 100.0
    x = Features(["tau", "moneyness"], market)(paths)
    assert x.shape == (3, 4, 2)
    assert torch.allclose(x[0, :, 0], torch.tensor([1.0, 0.75, 0.5, 0.25]))
    assert torch.all(x[..., 1] == 0)


def test_realized_vol_feature_is_normalized():
    market = MarketConfig(sigma_calm=0.1, sigma_crisis=0.5, p_crisis=0.2, vol_window=10)
    calib = regime_paths(market, 20000, np.random.default_rng(1))
    features = Features.calibrate(["tau", "moneyness", "realized_vol"], market, calib)
    rv = features(calib)[..., 2]
    assert abs(rv.mean().item()) < 1e-3 and abs(rv.std().item() - 1) < 1e-3


def test_oracle_delta_uses_regime_vol():
    market = MarketConfig(sigma_calm=0.1, sigma_crisis=0.5, p_crisis=1.0, vol_window=10)
    paths = regime_paths(market, 50, np.random.default_rng(2))
    delta = oracle_delta(paths, market)
    m, k = 0, paths.rupture[0].item()
    tau = (market.n_steps - k) * market.dt
    assert torch.isclose(delta[m, k], bs_delta(paths.S[m, k], market.K, 0.5, tau))


def test_bs_hedge_error_shrinks_with_rebalancing():
    """La couverture N(d1) laisse une erreur de discrétisation qui décroît quand on rebalance plus souvent."""
    S0, K, sigma, T = 100.0, 100.0, 0.2, 60 / 252
    p0 = bs_call_price(S0, K, sigma, T)
    losses = []
    for N in (15, 60):
        dt = T / N
        S = gbm_paths(20000, N, S0, 0.0, sigma, dt, np.random.default_rng(0))
        tau = (N - torch.arange(N, dtype=S.dtype)) * dt
        losses.append(mse_loss(call_pnl(S, bs_delta(S[:, :-1], K, sigma, tau), p0, K)).item())
    assert losses[1] < 0.5 * losses[0]
    assert math.isclose(losses[0] / losses[1], 4, rel_tol=0.3)    # variance en 1/N
