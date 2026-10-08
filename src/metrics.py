"""Mesures sur le jeu de test, toujours séparées entre régime calme et régime de crise.

 - loss, loss_calm, loss_crisis : perte sur toutes les trajectoires, les calmes, celles qui passent en crise ;
 - gap_calm   : moyenne de |delta - delta_oracle| sur les dates calmes (t < rupture) ;
 - gap_crisis : idem sur les dates de crise où l'information est disponible, t >= rupture + vol_window
                (avant, la volatilité réalisée n'a pas encore vu la crise : aucune couverture ne peut suivre l'oracle).
Une mesure sans date ou trajectoire concernée vaut NaN (par exemple les mesures de crise de E0).
"""

import math

import torch

from src.hedging import call_pnl


def _masked_mean(x, mask):
    return x[mask].mean().item() if mask.any() else math.nan


class Evaluator:
    """Précalcule tout ce qui ne dépend pas du réseau (entrées, delta oracle, masques), puis évalue vite."""

    def __init__(self, paths, features, delta_ref, p0, loss_fn, market):
        self.paths, self.p0, self.loss_fn, self.K = paths, p0, loss_fn, market.K
        self.inputs = features(paths)
        self.delta_ref = delta_ref
        t = torch.arange(market.n_steps)[None, :]
        self.crisis_paths = paths.crisis
        self.calm_dates = t < paths.rupture[:, None]
        self.informed_crisis_dates = t >= paths.rupture[:, None] + market.vol_window

    def losses(self, delta):
        """Pertes d'une couverture donnée (M, N), sur toutes les trajectoires puis par régime."""
        pnl = call_pnl(self.paths.S, delta, self.p0, self.K)
        out = {"loss": self.loss_fn(pnl).item()}
        for name, mask in [("calm", ~self.crisis_paths), ("crisis", self.crisis_paths)]:
            out[f"loss_{name}"] = self.loss_fn(pnl[mask]).item() if mask.any() else math.nan
        return out

    @torch.no_grad()
    def __call__(self, net):
        delta = net(self.inputs)
        gap = (delta - self.delta_ref).abs()
        return {**self.losses(delta),
                "gap_calm": _masked_mean(gap, self.calm_dates),
                "gap_crisis": _masked_mean(gap, self.informed_crisis_dates)}
