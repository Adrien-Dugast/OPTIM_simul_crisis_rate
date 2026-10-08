"""Lance une expérience et enregistre tout dans un nouveau dossier results/<expérience>/<date>_<étiquette>/.

Exemples (depuis la racine du dépôt) :
    python -m scripts.run e0
    python -m scripts.run e0bis --p_crisis 0.02 --seeds 0 1 2 --tag p002
    python -m scripts.run e0bis --learning_rates Adam=1e-2 Muon=1e-2 --n_iter 300 --tag test

Chaque champ de MarketConfig et TrainConfig (src/config.py) est une option ; les valeurs non données
gardent celles du préréglage.
"""

import argparse
import dataclasses
import time

import matplotlib

matplotlib.use("Agg")

from src.config import PRESETS, sections
from src.experiments import derived_values, prepare, run_all
from src.plots import make_figures
from src.runs import RunWriter, new_run_dir


def _parse_dict(items):
    return {k: float(v) for k, v in (item.split("=") for item in items)}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("preset", choices=list(PRESETS))
    parser.add_argument("--tag", help="suffixe du nom du dossier de résultats")
    parser.add_argument("--features", nargs="+", help="entrées du réseau : tau moneyness realized_vol")
    parser.add_argument("--no-figures", action="store_true")
    for section_name, section in sections(PRESETS["e0"]()).items():
        group = parser.add_argument_group(section_name)
        for f in dataclasses.fields(section):
            default = getattr(section, f.name)
            if isinstance(default, dict):
                group.add_argument(f"--{f.name}", nargs="+", metavar="NOM=LR", help="ex. Adam=1e-2 Muon=1e-2")
            elif isinstance(default, list):
                group.add_argument(f"--{f.name}", nargs="+", type=type(default[0]))
            else:
                group.add_argument(f"--{f.name}", type=type(default))
    return parser


def config_from_args(args):
    cfg = PRESETS[args.preset]()
    if args.features:
        cfg.features = args.features
    for section in sections(cfg).values():
        for f in dataclasses.fields(section):
            value = getattr(args, f.name)
            if value is not None:
                setattr(section, f.name, _parse_dict(value) if isinstance(getattr(section, f.name), dict) else value)
    return cfg


def main():
    args = build_parser().parse_args()
    cfg = config_from_args(args)
    if "realized_vol" in cfg.features and cfg.market.vol_window < 2:
        raise SystemExit("realized_vol demande --vol_window >= 2")

    run_dir = new_run_dir(cfg.experiment, args.tag)
    print(f"Run : {run_dir}", flush=True)
    t0 = time.time()
    setup = prepare(cfg)
    writer = RunWriter(run_dir)
    writer.save_config(cfg, derived_values(setup))
    refs = ", ".join(f"{k} {v['loss']:.4f}" for k, v in setup.references.items())
    print(f"p0 = {setup.p0:.4f} | pertes de référence : {refs}", flush=True)

    run_all(setup, writer)
    if not args.no_figures:
        make_figures(run_dir, setup)
    print(f"Terminé en {time.time() - t0:.0f} s : {run_dir}")


if __name__ == "__main__":
    main()
