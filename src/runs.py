"""Dossiers de résultats : un run = un dossier, jamais écrasé.

results/<expérience>/<date>_<heure>[_<étiquette>]/
    config.json   Config complète + valeurs dérivées (p0, normalisation, pertes de référence) + commit git
    history.csv   une ligne par (graine, optimiseur, point d'évaluation)
    models/       poids des réseaux entraînés, <optimiseur>_seed<graine>.pt
    figures/      figures PNG, regénérables par scripts/plot.py
"""

import csv
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import torch

from src.config import Config

RESULTS_DIR = Path("results")


def git_state():
    """Commit courant et présence de modifications non commitées, pour savoir avec quel code un run a été fait."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except OSError:
        return {"commit": None, "dirty": None}


def new_run_dir(experiment, tag=None):
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
    run_dir = RESULTS_DIR / experiment / name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "models").mkdir()
    (run_dir / "figures").mkdir()
    return run_dir


class RunWriter:
    """Écrit au fil de l'eau : chaque entraînement terminé est sur disque, même si le run s'interrompt ensuite."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.history_path = self.run_dir / "history.csv"

    def save_config(self, cfg, derived):
        cfg.save(self.run_dir / "config.json", derived=derived, git=git_state(),
                 started=datetime.now().isoformat(timespec="seconds"))

    def __call__(self, seed, name, net, history):
        torch.save(net.state_dict(), self.run_dir / "models" / f"{name}_seed{seed}.pt")
        rows = [{"seed": seed, "optimizer": name, **row} for row in history]
        new_file = not self.history_path.exists()
        with open(self.history_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new_file:
                writer.writeheader()
            writer.writerows(rows)


def load_config(run_dir):
    with open(Path(run_dir) / "config.json", encoding="utf-8") as f:
        d = json.load(f)
    return Config.from_dict(d), d["derived"]


def load_history(run_dir):
    """{optimiseur: {graine: {colonne: liste de valeurs}}}."""
    out = {}
    with open(Path(run_dir) / "history.csv", newline="") as f:
        for row in csv.DictReader(f):
            cols = out.setdefault(row.pop("optimizer"), {}).setdefault(int(row.pop("seed")), {})
            for key, value in row.items():
                cols.setdefault(key, []).append(float(value) if value != "" else math.nan)
    return out


def load_net(run_dir, name, seed, net):
    net.load_state_dict(torch.load(Path(run_dir) / "models" / f"{name}_seed{seed}.pt"))
    return net.eval()
