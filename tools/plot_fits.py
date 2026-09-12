#!/usr/bin/env python3
"""
Plot every fitted session's simulation against its measured data.

Reads outputs/result_solution_expN.csv from each session listed in
plot_fits.yaml and writes one figure inside each selected run directory. Sessions that have not been
fitted yet are skipped with a warning, so the config can list the whole set.

Usage:
    ./venv/bin/python3 tools/plot_fits.py
    ./venv/bin/python3 tools/plot_fits.py --config path/to/other.yaml
"""

import argparse
import glob
import logging
import os
import re
import sys
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "plot_fits.yaml")
sys.path.insert(0, REPO_ROOT)
from lib.utils.run_store import resolve_run

logger = logging.getLogger("plot_fits")


def setup_logging(log_file: str) -> None:
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)):
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(fmt)
        logger.addHandler(handler)


def experiment_files(outputs_dir: str) -> list[str]:
    """Return result_solution_expN.csv paths sorted by their numeric N."""
    paths = glob.glob(os.path.join(outputs_dir, "result_solution_exp*.csv"))

    def index(path: str) -> int:
        match = re.search(r"exp(\d+)\.csv$", path)
        return int(match.group(1)) if match else 0

    return sorted(paths, key=index)


def nrmse(measured: np.ndarray, simulated: np.ndarray) -> float:
    """Peak-normalised RMSE, as a percentage of the data's own range."""
    mask = np.isfinite(measured) & np.isfinite(simulated)
    measured, simulated = measured[mask], simulated[mask]
    if not measured.size:
        return float("nan")
    scale = np.max(np.abs(measured))
    if scale == 0:
        scale = 1.0
    return 100.0 * float(np.sqrt(np.mean((measured - simulated) ** 2))) / scale


def plot_session(session: dict, style: dict, out_dir: str, dpi: int) -> str | None:
    name = session["name"]
    root = os.path.join(REPO_ROOT, session["root"])
    outputs_dir = str(resolve_run(root, session.get("run")))
    out_dir = outputs_dir

    if not os.path.isdir(outputs_dir):
        logger.warning("%s: no outputs/ directory - not fitted yet, skipping", name)
        return None

    paths = experiment_files(outputs_dir)
    if not paths:
        logger.warning("%s: no result_solution_expN.csv in outputs/, skipping", name)
        return None

    panels = session["panels"]
    n_exp, n_panel = len(paths), len(panels)
    logger.info("%s: %d experiment(s) x %d panel(s)", name, n_exp, n_panel)

    # One column per panel, one row per experiment. Never a second y-axis on a
    # shared plot: two measures of different scale get their own panel.
    #
    # A single-panel session with many experiments is a small-multiples grid
    # rather than one tall column, so the conditions can be compared against
    # each other rather than scrolled past.
    if n_panel == 1 and n_exp > 3:
        n_cols = int(np.ceil(np.sqrt(n_exp)))
        n_rows = int(np.ceil(n_exp / n_cols))
    else:
        n_rows, n_cols = n_exp, n_panel

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 3.1 * n_rows),
        squeeze=False,
        facecolor=style["surface"],
    )
    grid = n_panel == 1 and n_exp > 3
    if grid:
        for spare in range(n_exp, n_rows * n_cols):
            axes[spare // n_cols][spare % n_cols].set_axis_off()

    for row, path in enumerate(paths):
        data = np.genfromtxt(path, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        t = data[:, 0]

        labels = session.get("experiment_labels") or []
        exp_label = labels[row] if row < len(labels) else f"experiment {row + 1}"

        for col, panel in enumerate(panels):
            ax = axes[row // n_cols][row % n_cols] if grid else axes[row][col]
            ax.set_facecolor(style["surface"])

            d_idx, s_idx = panel["data_col"], panel["sim_col"]
            # A panel may plot against a column other than time, and the
            # measured and simulated series may use DIFFERENT x columns — as in
            # a phase plot of one observable against another, where each series
            # carries its own abscissa. Defaults to time for both.
            xd_idx = panel.get("x_data_col", 0)
            xs_idx = panel.get("x_sim_col", 0)
            if max(d_idx, s_idx, xd_idx, xs_idx) >= data.shape[1]:
                logger.error(
                    "%s exp%d: panel '%s' wants columns %d/%d but the CSV has %d - "
                    "check the column mapping in the config",
                    name, row + 1, panel["label"], d_idx, s_idx, data.shape[1],
                )
                raise ValueError(f"Invalid plot column mapping for {name}: {panel}")

            measured, simulated = data[:, d_idx], data[:, s_idx]
            x_meas, x_sim = data[:, xd_idx], data[:, xs_idx]

            ax.plot(
                x_meas, measured, "o", markersize=5, color=style["color_data"],
                markeredgecolor=style["surface"], markeredgewidth=0.8,
                label="measured", zorder=3,
            )
            ax.plot(
                x_sim, simulated, "-", linewidth=2.0, color=style["color_fit"],
                label="fit", zorder=2,
            )

            if session.get("xscale") == "log" or panel.get("xscale") == "log":
                ax.set_xscale("log")
            if panel.get("yscale") == "log":
                # a log axis cannot show zero or negative values; clip to the
                # smallest positive value present so the axis limits stay honest
                ax.set_yscale("log")
                pos = np.concatenate([measured[measured > 0], simulated[simulated > 0]])
                if pos.size:
                    ax.set_ylim(bottom=0.5 * pos.min())

            err = nrmse(measured, simulated)
            logger.debug(
                "%s exp%d panel '%s': nRMSE %.2f%%", name, row + 1, panel["label"], err
            )

            # Text wears ink tokens, never the series color.
            ax.set_ylabel(panel["label"], fontsize=9, color=style["color_ink"])
            last_row = row >= n_exp - n_cols if grid else row == n_exp - 1
            # A panel with its own abscissa always needs its own label; a plain
            # time-series panel only labels the bottom row.
            if "xlabel" in panel:
                ax.set_xlabel(panel["xlabel"], fontsize=9, color=style["color_ink"])
            elif last_row:
                ax.set_xlabel(session.get("xlabel", "time"), fontsize=9,
                              color=style["color_ink"])
            title = f"{exp_label}  -  nRMSE {err:.2f}%" if n_panel == 1 else \
                    f"{exp_label} - {panel['label']}  -  nRMSE {err:.2f}%"
            ax.set_title(title, fontsize=9, color=style["color_ink_muted"], loc="left")

            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.tick_params(labelsize=8, colors=style["color_ink_muted"])
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(style["color_ink_muted"])
                ax.spines[spine].set_linewidth(0.8)

            # Two series, so a legend is mandatory - identity is never colour
            # alone. One legend for the figure is enough.
            if row == 0 and col == 0:
                ax.legend(fontsize=8, frameon=False, loc="best",
                          labelcolor=style["color_ink"])

    fig.suptitle(session.get("title", name), fontsize=12,
                 color=style["color_ink"], x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_fit.png")
    fig.savefig(out_path, dpi=dpi, facecolor=style["surface"])
    plt.close(fig)
    logger.info("%s: wrote %s", name, out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="path to the YAML config (default: beside this script)")
    parser.add_argument("--session", help="only plot this configured session")
    parser.add_argument("--run", help="run ID or directory; requires --session")
    parser.add_argument("--log-file", default=os.path.join(SCRIPT_DIR, "plot_fits.log"),
                        help="path to the log file")
    args = parser.parse_args()

    if args.run and not args.session:
        parser.error("--run requires --session")
    setup_logging(args.log_file)
    logger.info("reading config from %s", args.config)

    with open(args.config) as handle:
        config = yaml.safe_load(handle)

    out_dir = os.path.join(REPO_ROOT, config.get("output_dir", "outputs/fit_plots"))
    dpi = config.get("dpi", 140)
    style = config["style"]

    written, skipped = [], []
    selected = [s for s in config["sessions"] if not args.session or s["name"] == args.session]
    if not selected:
        logger.error("No matching session in plotting config")
        return 1
    for session in selected:
        if args.run:
            session = dict(session, run=args.run)
        try:
            path = plot_session(session, style, out_dir, dpi)
            if path:
                destination = Path(path).parent
                (destination / "plot_fits.yaml").write_text(yaml.safe_dump(
                    dict(config, sessions=[session]), sort_keys=False))
                script_copy = destination / "plot_fits_script.py"
                if Path(__file__).resolve() != script_copy.resolve():
                    shutil.copyfile(__file__, script_copy)
        except Exception:
            logger.exception("%s: failed to plot", session.get("name", "?"))
            skipped.append(session.get("name", "?"))
            continue
        (written if path else skipped).append(session.get("name", "?"))

    logger.info("plotted %d session(s): %s", len(written), ", ".join(written) or "none")
    if skipped:
        logger.warning("skipped %d session(s): %s", len(skipped), ", ".join(skipped))
    logger.info("figures saved in the selected run directories")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
