"""Configuration d'une expérience : marché simulé, entrées du réseau, entraînement.

Une expérience = une Config. Les préréglages (PRESETS) donnent E0 et E0 bis ; tout champ se modifie
en ligne de commande (voir scripts/run.py), et la Config complète est écrite dans le dossier du run.
"""

import dataclasses
import json
from dataclasses import dataclass, field


@dataclass
class MarketConfig:
    S0: float = 100.0
    K: float = 100.0
    n_steps: int = 60               # maturité en jours = nombre de dates de rebalancement
    dt: float = 1 / 252             # un jour ouvré, en années (r = 0 partout)
    sigma_calm: float = 0.2
    sigma_crisis: float = 0.5
    p_crisis: float = 0.0           # proportion des trajectoires avec rupture de volatilité
    vol_window: int = 0             # fenêtre de la volatilité réalisée, en jours (0 : pas de pré-historique)

    @property
    def T(self):
        return self.n_steps * self.dt


@dataclass
class TrainConfig:
    learning_rates: dict = field(default_factory=lambda: {"GD": 1e-3, "Adam": 1e-2, "Muon": 1e-2, "Spectral": 1e-2})
    lr_biases: float = 3e-3         # Adam sur les biais, pour Muon et Spectral
    n_iter: int = 1000
    batch_size: int = 2048
    eval_every: int = 50
    seeds: list = field(default_factory=lambda: [0])
    width: int = 64
    loss: str = "mse"               # clé de hedging.LOSSES
    n_test: int = 20000
    test_seed: int = 12345
    n_calib: int = 200000           # échantillon de calibration : prime p0 et normalisation des entrées
    calib_seed: int = 999


@dataclass
class Config:
    experiment: str
    features: list                  # parmi "tau", "moneyness", "realized_vol"
    market: MarketConfig
    train: TrainConfig

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(experiment=d["experiment"], features=list(d["features"]),
                   market=MarketConfig(**d["market"]), train=TrainConfig(**d["train"]))

    def save(self, path, **extra):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({**self.to_dict(), **extra}, f, indent=2, ensure_ascii=False)


PRESETS = {
    # E0 : contrôle, Black-Scholes sans crise
    "e0": lambda: Config(
        experiment="e0",
        features=["tau", "moneyness"],
        market=MarketConfig(sigma_calm=0.2, p_crisis=0.0, vol_window=0),
        train=TrainConfig(),
    ),
    # E0 bis : rupture de volatilité 0.1 -> 0.5 sur une proportion p des trajectoires,
    # le réseau voit la volatilité réalisée sur vol_window jours
    "e0bis": lambda: Config(
        experiment="e0bis",
        features=["tau", "moneyness", "realized_vol"],
        market=MarketConfig(sigma_calm=0.1, sigma_crisis=0.5, p_crisis=0.05, vol_window=10),
        train=TrainConfig(),
    ),
}


def sections(cfg):
    """Les sous-configurations dont les champs sont modifiables en ligne de commande."""
    return {"market": cfg.market, "train": cfg.train}
