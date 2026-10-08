"""Préparation d'une expérience, boucle d'entraînement et comparaison des optimiseurs.

Le déroulé ne dépend que de la Config : on peut donc tout reconstruire (jeu de test, entrées, références)
à partir du config.json d'un run, par exemple pour refaire les figures.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch

from src.hedging import LOSSES, Features, HedgeNet, bs_call_price, call_pnl, oracle_delta, realized_vol_delta
from src.metrics import Evaluator
from src.optimizers import make_optimizer
from src.simulation import regime_paths


@dataclass
class Setup:
    """Tout ce qui est commun aux optimiseurs et aux graines d'une expérience."""
    cfg: object
    features: Features
    p0: float
    evaluator: Evaluator
    references: dict        # perte des couvertures de référence sur le jeu de test


def premium(cfg, calib_paths):
    """Prime de vente p0. Avec r = 0, S est une martingale : E[gains] = 0 pour toute couverture, donc le p0
    qui minimise E[P&L^2] est E[payoff]. Sans crise c'est le prix BS ; avec crise, estimé par Monte-Carlo."""
    m = cfg.market
    if m.p_crisis == 0:
        return bs_call_price(m.S0, m.K, m.sigma_calm, m.T)
    return torch.clamp(calib_paths.S[:, -1] - m.K, min=0).mean().item()


def prepare(cfg, derived=None):
    """Construit le Setup. derived : valeurs déjà calculées (p0, normalisation) lues dans un run existant."""
    m = cfg.market
    if derived is None:
        calib = regime_paths(m, cfg.train.n_calib, np.random.default_rng(cfg.train.calib_seed))
        features, p0 = Features.calibrate(cfg.features, m, calib), premium(cfg, calib)
    else:
        features, p0 = Features(cfg.features, m, **derived["features"]), derived["p0"]

    test = regime_paths(m, cfg.train.n_test, np.random.default_rng(cfg.train.test_seed))
    loss_fn = LOSSES[cfg.train.loss]
    evaluator = Evaluator(test, features, oracle_delta(test, m), p0, loss_fn, m)
    references = {"oracle": evaluator.losses(evaluator.delta_ref)}
    if m.vol_window > 0:
        references["realized_vol"] = evaluator.losses(realized_vol_delta(test, m))
    return Setup(cfg, features, p0, evaluator, references)


def derived_values(setup):
    """Ce qu'on écrit dans config.json en plus de la Config : prime, normalisation, références."""
    return {"p0": setup.p0, "features": setup.features.state(), "references": setup.references}


def make_net(setup):
    return HedgeNet(n_inputs=len(setup.features), width=setup.cfg.train.width)


def train(net, optimizer, setup, rng, log=None):
    """Une itération = un nouveau lot, une perte sur le P&L, un pas d'optimiseur.
    Renvoie l'historique : une ligne (dict) par point d'évaluation."""
    cfg, loss_fn = setup.cfg, LOSSES[setup.cfg.train.loss]
    history = []
    t_start = time.time()
    for it in range(cfg.train.n_iter):
        batch = regime_paths(cfg.market, cfg.train.batch_size, rng)
        pnl = call_pnl(batch.S, net(setup.features(batch)), setup.p0, cfg.market.K)
        loss = loss_fn(pnl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % cfg.train.eval_every == 0 or it == cfg.train.n_iter - 1:
            row = {"iteration": it, "time_s": time.time() - t_start, "train_loss": loss.item(), **setup.evaluator(net)}
            history.append(row)
            if log:
                log(row)
    return history


def run_all(setup, on_trained, verbose=True):
    """Entraîne un réseau par (graine, optimiseur), avec la même initialisation et les mêmes lots pour une graine.
    on_trained(seed, name, net, history) est appelé après chaque entraînement (sauvegarde)."""
    tc = setup.cfg.train
    for seed in tc.seeds:
        for name, lr in tc.learning_rates.items():
            torch.manual_seed(seed)                   # même initialisation du réseau
            rng = np.random.default_rng(seed)         # mêmes lots de trajectoires
            net = make_net(setup)
            optimizer = make_optimizer(name, net, lr, lr_biases=tc.lr_biases)
            history = train(net, optimizer, setup, rng)
            on_trained(seed, name, net, history)
            if verbose:
                last = history[-1]
                print(f"graine {seed} | {name:9s} lr = {lr:<7} | perte test {last['loss']:.4f} "
                      f"(calme {last['loss_calm']:.4f}, crise {last['loss_crisis']:.4f}) | {last['time_s']:.0f} s",
                      flush=True)
