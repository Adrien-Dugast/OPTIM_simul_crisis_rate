"""Construction des optimiseurs comparés.

 - GD       : descente de gradient classique (sur mini-lots), W <- W - lr * G
 - Adam     : préconditionneur diagonal, chaque coordonnée normalisée par sa propre échelle
 - Spectral : descente la plus raide en norme spectrale, exacte par SVD : G = U S V^T, W <- W - lr * U V^T
              (toutes les valeurs singulières du pas ramenées à 1, sans momentum)
 - Muon     : momentum + même orthogonalisation, approchée par Newton-Schulz (torch.optim.Muon)
Spectral et Muon ne s'appliquent qu'aux matrices de poids (2D) ; les biais sont confiés à Adam,
comme le recommande Muon. GD et Adam traitent tous les paramètres.
"""

import math

import torch


class SpectralGD(torch.optim.Optimizer):
    """Spectral GD exact : W <- W - lr * sqrt(max(1, A/B)) * U V^T, avec G = U S V^T (SVD réduite).
    Le facteur sqrt(max(1, A/B)) est le même que celui de torch.optim.Muon, pour que les deux soient comparables."""

    def __init__(self, params, lr):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                U, _, Vh = torch.linalg.svd(p.grad, full_matrices=False)
                A, B = p.shape
                p.add_(U @ Vh, alpha=-group["lr"] * math.sqrt(max(1, A / B)))


class CombinedOptimizer:
    """Plusieurs optimiseurs sur des groupes de paramètres disjoints, utilisés comme un seul."""

    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


def make_optimizer(name, net, lr, lr_biases=3e-3):
    """name parmi "GD", "Adam", "Spectral", "Muon"."""
    matrices = [p for p in net.parameters() if p.ndim == 2]     # poids des couches Linear
    biases = [p for p in net.parameters() if p.ndim != 2]
    if name == "GD":
        return torch.optim.SGD(net.parameters(), lr=lr)
    if name == "Adam":
        return torch.optim.Adam(net.parameters(), lr=lr)
    if name == "Spectral":
        return CombinedOptimizer(SpectralGD(matrices, lr=lr), torch.optim.Adam(biases, lr=lr_biases))
    if name == "Muon":
        return CombinedOptimizer(torch.optim.Muon(matrices, lr=lr, weight_decay=0.0),   # weight_decay=0.1 par défaut
                                 torch.optim.Adam(biases, lr=lr_biases))
    raise ValueError(f"optimiseur inconnu : {name}")
