#!/usr/bin/env python3
"""Plot the hif1a_1pct_fit result: model vs. data for each observable.

Reads outputs/result_solution_exp1.csv (written by fit_parameters.py) and
overlays the fitted model trajectory on the measured data for every observable
listed in the config. Configuration lives in plot_fit_config.yaml next to this
script (in the session root, so it survives fit_parameters.py clearing outputs/);
the only CLI arg is an optional --config pointing to an alternate config path.
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
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)):
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(fmt)
        root.addHandler(handler)
    # matplotlib's font manager is extremely chatty at DEBUG; keep our own logs clean.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot model vs data fit for hif1a_1pct_fit.")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "plot_fit_config.yaml"),
                        help="Path to YAML config (default: alongside this script).")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    setup_logging(SCRIPT_DIR / "plot_fit.log")
    logging.info("plotting fit using config %s", config_path)

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    base = config_path.parent
    result_csv = base / cfg["result_csv"]
    output_png = base / cfg["output_png"]
    logging.debug("reading result CSV %s", result_csv)

    data = np.loadtxt(result_csv, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    logging.info("loaded %d time points, %d columns", data.shape[0], data.shape[1])

    t = data[:, 0] / float(cfg["time_divisor"])
    observables = cfg["observables"]
    n = len(observables)

    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 4.6), squeeze=False)
    for ax, obs in zip(axes[0], observables):
        d = data[:, obs["data_col"]]
        m = data[:, obs["model_col"]]
        rmse = float(np.sqrt(np.mean((m - d) ** 2)))
        ax.plot(t, d, "o", color="#1b1b1b", markersize=7, label="data", zorder=3)
        ax.plot(t, m, "-", color="#c0392b", linewidth=2.2, marker="s", markersize=4,
                label="fit", zorder=2)
        ax.set_title(f"{obs['title']}  (RMSE={rmse:.4f})")
        ax.set_xlabel(cfg["time_label"])
        ax.set_ylabel(obs["ylabel"])
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        logging.info("%s: RMSE=%.5f", obs["title"], rmse)

    loss = cfg.get("final_loss")
    suptitle = "hif1a_1pct_fit — fitted model vs. data"
    if loss is not None:
        suptitle += f"  (final loss {float(loss):.4f})"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_png, dpi=150)
    logging.info("wrote %s", output_png)


if __name__ == "__main__":
    main()
