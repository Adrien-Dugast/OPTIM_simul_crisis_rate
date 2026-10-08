import math

import torch

from src.optimizers import SpectralGD


def _step(opt_class, G, **kwargs):
    """Applique un pas de l'optimiseur à W = 0 de gradient G et renvoie le pas effectué."""
    W = torch.nn.Parameter(torch.zeros_like(G))
    W.grad = G.clone()
    opt_class([W], **kwargs).step()
    return -W.detach()


def test_spectral_step_is_UVt():
    torch.manual_seed(0)
    G = torch.randn(8, 5)
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    step = _step(SpectralGD, G, lr=1.0)
    assert torch.allclose(step, math.sqrt(8 / 5) * U @ Vh, atol=1e-5)


def test_muon_close_to_exact_orthogonalization():
    """Newton-Schulz approche U V^T : toutes les valeurs singulières du pas proches de 1, même direction."""
    torch.manual_seed(0)
    G = torch.randn(32, 32)
    G = G @ torch.diag(torch.logspace(0, -1, 32))    # gradient modérément mal conditionné
    U, _, Vh = torch.linalg.svd(G)
    step = _step(torch.optim.Muon, G, lr=1.0, momentum=0.0, weight_decay=0.0)
    s = torch.linalg.svdvals(step)
    # Newton-Schulz de Muon : valeurs dans [~0.7, ~1.2]. Avec un conditionnement de 1e-2 au lieu de 1e-1,
    # les plus petites restent nettement sous 1 (0.23 observé) : 5 itérations ne suffisent pas à les remonter.
    assert s.max() < 1.3 and s.min() > 0.5
    cos = (step * (U @ Vh)).sum() / (step.norm() * (U @ Vh).norm())
    assert cos > 0.9
