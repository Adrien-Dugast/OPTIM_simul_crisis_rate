"""Regénère les figures d'un ou plusieurs runs à partir de leur dossier, sans réentraîner.

    python -m scripts.plot results/e0bis/20261008_153000_p005 [autres dossiers...]
"""

import argparse

import matplotlib

matplotlib.use("Agg")

from src.plots import make_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    for run_dir in parser.parse_args().run_dirs:
        names = make_figures(run_dir)
        print(f"{run_dir} : figures {', '.join(names)}")


if __name__ == "__main__":
    main()
