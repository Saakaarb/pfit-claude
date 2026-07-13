#!/usr/bin/env python3
"""Plot model vs. data from outputs/result_solution_exp1.csv for this example.

Config lives in plot_fit_config.yaml next to this script (kept in the example
root because fit_parameters.py clears outputs/ on each run). Only CLI arg is an
optional --config.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def setup_logging(log_file: Path) -> None:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)):
        h.setLevel(logging.DEBUG)
        h.setFormatter(fmt)
        root.addHandler(h)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot fit vs data.")
    ap.add_argument("--config", default=str(SCRIPT_DIR / "plot_fit_config.yaml"))
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    setup_logging(SCRIPT_DIR / "plot_fit.log")
    logging.info("plotting using %s", cfg_path)

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    base = cfg_path.parent
    data = np.loadtxt(base / cfg["result_csv"], delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    logging.info("loaded %d rows, %d cols", *data.shape)

    t = data[:, 0] / float(cfg.get("time_divisor", 1.0))
    logx = bool(cfg.get("logx", False))
    obs = cfg["observables"]
    fig, axes = plt.subplots(1, len(obs), figsize=(5.6 * len(obs), 4.4), squeeze=False)
    for ax, o in zip(axes[0], obs):
        d = data[:, o["data_col"]]
        m = data[:, o["model_col"]]
        rmse = float(np.sqrt(np.mean((m - d) ** 2)))
        plot = ax.semilogx if logx else ax.plot
        plot(t, d, "o", color="#1b1b1b", markersize=5, label="data", zorder=3)
        plot(t, m, "-", color="#c0392b", linewidth=2.0, label="fit", zorder=2)
        ax.set_title(f"{o['title']}  (RMSE={rmse:.3g})")
        ax.set_xlabel(cfg.get("time_label", "time"))
        ax.set_ylabel(o["ylabel"])
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        logging.info("%s RMSE=%.4g", o["title"], rmse)

    if cfg.get("suptitle"):
        fig.suptitle(cfg["suptitle"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95) if cfg.get("suptitle") else None)
    fig.savefig(base / cfg["output_png"], dpi=150)
    logging.info("wrote %s", base / cfg["output_png"])


if __name__ == "__main__":
    main()
