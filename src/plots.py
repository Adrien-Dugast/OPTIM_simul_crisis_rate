"""Figures d'un run, construites uniquement à partir de son dossier (config.json, history.csv, models/).

Courbes : moyenne sur les graines, bande = ± un écart-type.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from src.experiments import make_net, prepare
from src.hedging import bs_delta
from src.runs import load_config, load_history, load_net

COLORS = {"GD": "#2a78d6", "Adam": "#eb6834", "Muon": "#1baf7a", "Spectral": "#e87ba4"}
CURVE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]
REF_STYLES = {"oracle": ("N(d1), régime connu (oracle)", "--"), "realized_vol": ("N(d1) avec vol. réalisée", ":")}


def _clean(ax):
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def _mean_std(seeds, key):
    values = np.array([cols[key] for cols in seeds.values()])
    return values.mean(axis=0), values.std(axis=0)


def _plot_metric(ax, history, key, x="iteration"):
    for name, seeds in history.items():
        xs, _ = _mean_std(seeds, x)
        mean, std = _mean_std(seeds, key)
        ax.plot(xs, mean, color=COLORS.get(name), lw=2, label=name)
        if len(seeds) > 1:
            ax.fill_between(xs, mean - std, mean + std, color=COLORS.get(name), alpha=0.2, lw=0)


def _has_values(history, key):
    return any(not all(math.isnan(v) for v in cols[key]) for seeds in history.values() for cols in seeds.values())


def plot_losses(history, references, title):
    """Perte de test totale en fonction des itérations et du temps de calcul, avec les couvertures de référence."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, x, xlabel in zip(axes, ["iteration", "time_s"], ["itération", "temps de calcul (s)"]):
        _plot_metric(ax, history, "loss", x)
        for ref, losses in references.items():
            label, ls = REF_STYLES[ref]
            ax.axhline(losses["loss"], color="gray", ls=ls, lw=1.5, label=label)
        ax.set_xlabel(xlabel)
        ax.set_yscale("log")
        _clean(ax)
    axes[0].set_ylabel("perte de test")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_by_regime(history, references, title):
    """Perte et écart au delta oracle, séparés entre régime calme et crise, en fonction des itérations."""
    panels = [("loss_calm", "perte, trajectoires calmes", True), ("loss_crisis", "perte, trajectoires de crise", True),
              ("gap_calm", "|delta - oracle|, dates calmes", False),
              ("gap_crisis", "|delta - oracle|, crise (après la fenêtre)", False)]
    panels = [p for p in panels if _has_values(history, p[0])]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.8))
    for ax, (key, label, is_loss) in zip(np.atleast_1d(axes), panels):
        _plot_metric(ax, history, key)
        if is_loss:
            for ref, losses in references.items():
                ref_label, ls = REF_STYLES[ref]
                ax.axhline(losses[key], color="gray", ls=ls, lw=1.5, label=ref_label)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("itération")
        ax.set_yscale("log")
        _clean(ax)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def delta_scenarios(cfg):
    """Coupes tracées : (étiquette, date t, vol. réalisée donnée au réseau, vol. de la référence N(d1))."""
    m = cfg.market
    if "realized_vol" not in cfg.features:
        return [(f"t = {t} j", t, None, m.sigma_calm) for t in (0, m.n_steps // 2, m.n_steps - 5)]
    t = m.n_steps // 2
    return [(f"calme, vol. réalisée {m.sigma_calm:g}", t, m.sigma_calm, m.sigma_calm),
            (f"crise, vol. réalisée {m.sigma_crisis:g}", t, m.sigma_crisis, m.sigma_crisis)]


def plot_deltas(nets, setup, title):
    """Delta appris (trait plein) vs N(d1) (pointillés), un panneau par optimiseur, plus la référence seule."""
    cfg = setup.cfg
    m = cfg.market
    width = 0.3 if m.p_crisis > 0 else 0.15
    S_grid = torch.linspace(m.K * (1 - width), m.K * (1 + width), 200)
    scenarios = delta_scenarios(cfg)
    panels = list(nets.items()) + [("Référence : N(d1)", None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.5), sharey=True)
    for ax, (name, net) in zip(axes, panels):
        for (label, t, sigma_hat, sigma_ref), c in zip(scenarios, CURVE_COLORS):
            ref = bs_delta(S_grid, m.K, sigma_ref, (m.n_steps - t) * m.dt)
            if net is None:
                ax.plot(S_grid, ref, color=c, lw=2, label=label)
                continue
            with torch.no_grad():
                ax.plot(S_grid, net(setup.features.on_grid(S_grid, t, sigma_hat)), color=c, lw=2, label=label)
            ax.plot(S_grid, ref, color=c, lw=1.5, ls="--")
        ax.set_title(name)
        ax.set_xlabel("prix S_t")
        _clean(ax)
    axes[0].set_ylabel("delta_t")
    handles, _ = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color="gray", lw=2, label="réseau"),
                Line2D([], [], color="gray", lw=1.5, ls="--", label="N(d1)")]
    axes[0].legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    if "realized_vol" in cfg.features:
        title += f" (t = {scenarios[0][1]} j)"
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def make_figures(run_dir, setup=None):
    """Regénère toutes les figures d'un run dans run_dir/figures/. setup : évite de le reconstruire s'il existe déjà."""
    run_dir = Path(run_dir)
    cfg, derived = load_config(run_dir)
    if setup is None:
        setup = prepare(cfg, derived)
    history = load_history(run_dir)
    references = derived["references"]
    m = cfg.market
    title = f"{cfg.experiment} : p = {m.p_crisis:g}, vol. {m.sigma_calm:g}" + \
            (f" -> {m.sigma_crisis:g}" if m.p_crisis > 0 else "") + f", {len(cfg.train.seeds)} graine(s)"

    first_seed = cfg.train.seeds[0]
    nets = {name: load_net(run_dir, name, first_seed, make_net(setup)) for name in history}
    figures = {"loss": plot_losses(history, references, title),
               "by_regime": plot_by_regime(history, references, title),
               "deltas": plot_deltas(nets, setup, f"Delta appris, graine {first_seed}")}
    for name, fig in figures.items():
        fig.savefig(run_dir / "figures" / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    return list(figures)
