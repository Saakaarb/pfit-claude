#!/usr/bin/env python3
"""
Plot the loss trajectory and the fitted model for each configured session.

Two figures per session, written into that session's outputs/ directory:

  loss_curves.png  the population and gradient stages, on a shared loss axis
  fit_result.png   the fitted model against the measured data

The model curve is produced by re-integrating user_model.py at the fitted
parameters on a dense time grid, rather than by joining the stored
result_solution_expN.csv rows. Those rows exist only at the data's own sample
times, which for a sparse or spiky record draws a polyline that misrepresents
the trajectory between samples.

Usage:
    ./venv/bin/python3 tools/plot_diagnostics.py
    ./venv/bin/python3 tools/plot_diagnostics.py --config path/to/other.yaml
"""

import argparse
import importlib.util
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
from scipy.integrate import solve_ivp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "plot_diagnostics.yaml")
sys.path.insert(0, REPO_ROOT)
from lib.utils.run_store import resolve_run, run_sources, run_config
from lib.utils.dataset_io import load_dataset

logger = logging.getLogger("plot_diagnostics")

ITER_LINE = re.compile(r"^\s*(\d+)\s*,\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*$")


def setup_logging(log_file):
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)):
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def read_iteration_log(path):
    """Return (iteration, best_cost) arrays from an optimizer log."""
    iters, costs = [], []
    with open(path) as fh:
        for line in fh:
            m = ITER_LINE.match(line)
            if m:
                iters.append(int(m.group(1)))
                costs.append(float(m.group(2)))
    return np.array(iters), np.array(costs)


def load_user_model(session_dir):
    """Import the session's user_model.py as a module."""
    path = os.path.join(session_dir, "generated", "user_model.py")
    spec = importlib.util.spec_from_file_location("user_model_plot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dense_solution(cfg, scfg, session_dir, config):
    """Re-integrate the selected run's model at its fitted parameters."""
    output = resolve_run(session_dir, scfg.get("run"))
    sources = run_sources(output, session_dir)
    with open(run_config(output, session_dir)) as fh:
        uin = yaml.safe_load(fh)

    names = [p["name"] for p in uin["model"]["trainable_parameters"]]
    fitted = np.loadtxt(
        os.path.join(output, "final_design_point.csv")
    )
    fitted = np.atleast_1d(fitted)
    trainable = dict(zip(names, fitted))
    logger.debug("fitted parameters: %s", trainable)

    fixed = {p["name"]: float(p["value"])
             for p in uin["model"].get("fixed_parameters", []) or []}
    y0 = np.array([float(v["init_val"])
                   for v in uin["model"]["integrated_variables"]])

    data_file = uin["experiments"][0]["data_file"]
    data = load_dataset(Path(sources) / "inputs" / data_file)
    t_eval = data[:, 0]

    init_time = float(uin["gradient_opt"].get("initial_time", t_eval[0]))
    t0 = init_time if scfg.get("dense_from_initial_time") else t_eval[0]
    t1 = float(t_eval[-1])

    model = load_user_model(sources)

    def rhs(t, y):
        return np.asarray(
            model.user_defined_system(t, y, trainable, fixed, data[:, 1:], t_eval),
            dtype=float,
        )

    grid = np.linspace(t0, t1, int(config["dense_points"]))
    logger.info("re-integrating over [%g, %g] at %d points", t0, t1, grid.size)
    sol = solve_ivp(
        rhs, (t0, t1), y0, t_eval=grid,
        method=config["dense_solver"],
        rtol=float(config["dense_rtol"]), atol=float(config["dense_atol"]),
    )
    if not sol.success:
        logger.error("dense re-integration failed: %s", sol.message)
        return None, None, data
    logger.info("dense re-integration OK (%d states)", sol.y.shape[0])
    return sol.t, sol.y, data


def plot_losses(session_dir, scfg, config, out_path):
    """Both stages on a shared loss axis, one panel each."""
    pal = config["palette"]
    outputs = str(resolve_run(session_dir, scfg.get("run")))

    stages = []
    for fname, label, color in (
        ("de_fitting.log", "Stage 1: differential evolution", pal["stage_1"]),
        ("pso_fitting.log", "Stage 1: particle swarm", pal["stage_1"]),
        ("NODE_fitting.log", "Stage 2: gradient refinement (adam)", pal["stage_2"]),
    ):
        path = os.path.join(outputs, fname)
        if not os.path.exists(path):
            continue
        it, cost = read_iteration_log(path)
        if it.size:
            stages.append((label, color, it, cost))
            logger.info("%s: %d iterations, %.4g -> %.4g",
                        fname, it.size, cost[0], cost[-1])

    if not stages:
        logger.warning("no optimizer logs with iterations; skipping loss figure")
        return

    fig, axes = plt.subplots(
        1, len(stages), figsize=(6.2 * len(stages), 4.4), squeeze=False, sharey=True
    )
    lo = min(c.min() for _, _, _, c in stages)
    hi = max(c.max() for _, _, _, c in stages)

    for ax, (label, color, it, cost) in zip(axes[0], stages):
        ax.plot(it, cost, color=color, linewidth=2.0, solid_capstyle="round")
        ax.set_title(label, fontsize=11, loc="left")
        ax.set_xlabel("iteration")
        ax.set_yscale("log")
        ax.set_ylim(lo * 0.85, hi * 1.15)
        ax.grid(True, which="major", color=config["palette"]["grid"],
                linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # Direct label on the final value, so the endpoint is readable without
        # cross-referencing the axis.
        ax.annotate(f"{cost[-1]:.3f}", xy=(it[-1], cost[-1]),
                    xytext=(-4, 8), textcoords="offset points",
                    ha="right", fontsize=10, color=color, fontweight="bold")
        ax.annotate(f"{cost[0]:.3f}", xy=(it[0], cost[0]),
                    xytext=(4, 6), textcoords="offset points",
                    ha="left", fontsize=9, color=pal["text_secondary"])

    axes[0][0].set_ylabel(scfg.get("loss_label", "loss"))
    fig.suptitle(scfg["title"] + "  -  loss trajectory", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def stored_solution(session_dir, scfg):
    """Read the framework's own result_solution_exp1.csv instead of re-solving.

    Correct choice whenever the data grid is already dense: the stored file was
    produced by the same diffrax solve that computed the loss, so it cannot
    disagree with the reported number. Re-integrating with scipy can, and for a
    stiff regularised RHS driven by a 24841-point interpolated input it is both
    slow and unreliable.

    Each series names `model_col`, an index into that file, instead of `state`.
    """
    path = os.path.join(resolve_run(session_dir, scfg.get("run")), "result_solution_exp1.csv")
    stored = np.loadtxt(path, delimiter=",")
    logger.info("using stored solution %s (%d rows, %d cols)",
                path, *stored.shape)
    return stored[:, 0], stored


def plot_fit(session_dir, scfg, config, out_path):
    """Fitted model against the measured data."""
    pal = config["palette"]
    if scfg.get("use_stored_solution"):
        t, stored = stored_solution(session_dir, scfg)
        output = resolve_run(session_dir, scfg.get("run"))
        sources = run_sources(output, session_dir)
        with open(run_config(output, session_dir)) as handle:
            uin = yaml.safe_load(handle)
        data = load_dataset(Path(sources) / "inputs" / uin["experiments"][0]["data_file"])
        # Present the stored model columns under the same interface the panels
        # already use, so `state: n` indexes series n of the stored file.
        ys = {i: stored[:, i] for i in range(stored.shape[1])}
    else:
        t, ys, data = dense_solution(config, scfg, session_dir, config)
        if t is None:
            logger.error("skipping fit figure for %s", scfg["name"])
            return

    panels = scfg["panels"]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(7.0 * len(panels), 4.8), squeeze=False
    )
    t_data = data[:, 0]

    for ax, panel in zip(axes[0], panels):
        n_series = len(panel["series"])
        for s in panel["series"]:
            color = pal[s["color"]]
            # `state: null` marks a data-only series - an exogenous input, which
            # is recorded but is not a state the model integrates. A residual
            # panel plots only the difference, never the trajectory itself.
            if s.get("state") is not None and not panel.get("residual"):
                ax.plot(t, ys[s["state"]], color=color, linewidth=2.0,
                        solid_capstyle="round", zorder=3,
                        label=s["label"] if n_series > 1 else None)

            # A residual panel. When the fit is tight the model and the data
            # overlap into one line and the figure says nothing about quality;
            # the residual is the only view that does.
            if panel.get("residual") and s.get("data_col") is not None:
                col = data[:, int(s["data_col"])]
                m = ~np.isnan(col)
                sim = np.interp(t_data[m], t, ys[s["state"]])
                ax.axhline(0.0, color=pal["grid"], linewidth=1.0, zorder=1)
                ax.plot(t_data[m], sim - col[m], color=color, linewidth=1.2,
                        zorder=3, label=s["label"] if n_series > 1 else None)
                continue

            if s.get("data_col") is not None:
                col = data[:, int(s["data_col"])]
                m = ~np.isnan(col)
                dense = m.sum() > 40
                # Error bars are per-point information; on a densely sampled
                # record they overlap into a solid block and stop carrying any.
                # A constant sigma is stated in the caption instead.
                show_err = s.get("errorbars", "auto")
                draw_err = show_err is True or (show_err == "auto" and not dense)
                err = (data[:, int(s["sigma_col"])][m]
                       if (draw_err and s.get("sigma_col") is not None) else None)
                ax.errorbar(
                    t_data[m], col[m], yerr=err, fmt="o",
                    markersize=3.0 if dense else 6.0,
                    markerfacecolor="white" if not dense else color,
                    markeredgecolor=color, markeredgewidth=1.6 if not dense else 0.0,
                    ecolor=color, elinewidth=1.0, capsize=0,
                    alpha=0.55 if dense else 1.0,
                    linestyle="none", zorder=4,
                )

            # Direct label at the curve's end: the relief rule, since two of
            # the four categorical slots sit below 3:1 contrast on white.
            # Suppressed where curves converge at the right edge and the labels
            # would collide - there the legend carries identity instead, which
            # is only safe for slots that clear 3:1 on their own.
            if s.get("state") is None:
                continue
            finite = np.isfinite(ys[s["state"]])
            if panel.get("direct_labels", True) and finite.any():
                idx = np.where(finite)[0][-1]
                ax.annotate(s["label"], xy=(t[idx], ys[s["state"]][idx]),
                            xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=9.5, color=color,
                            fontweight="bold", annotation_clip=False)

        ax.set_title(panel["title"], fontsize=11, loc="left")
        ax.set_xlabel(scfg.get("time_label", "time"))
        ax.set_ylabel(panel["ylabel"])
        if panel.get("yscale") == "log":
            ax.set_yscale("log")
        # Without a limit the axis follows the model wherever it goes, which on
        # a log scale can be many decades below anything that was measured -
        # squashing the region the fit is actually judged on.
        if panel.get("ylim"):
            ax.set_ylim(float(panel["ylim"][0]), float(panel["ylim"][1]))
        ax.grid(True, which="major", color=pal["grid"], linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(x=0.13)
        if n_series > 1:
            ax.legend(frameon=False, fontsize=9, loc="best")

    # Default caption assumes error bars; a record with no uncertainty column
    # must not claim them, so a session can override it.
    caption = scfg.get("caption",
                       "markers = measurements +/- 1 sigma, line = fit")
    fig.suptitle(scfg["title"] + "  -  fitted model vs data   (" + caption + ")",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--session")
    parser.add_argument("--run", help="run ID or path; requires --session")
    parser.add_argument(
        "--log-file", default=os.path.join(SCRIPT_DIR, "plot_diagnostics.log")
    )
    args = parser.parse_args()

    if args.run and not args.session:
        parser.error("--run requires --session")
    setup_logging(args.log_file)
    logger.info("reading config %s", args.config)
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    for scfg in config["sessions"]:
        name = scfg["name"]
        if args.session and name != args.session:
            continue
        if args.run:
            scfg = dict(scfg, run=args.run)
        session_dir = os.path.join(REPO_ROOT, "sessions", name)
        try:
            outputs = str(resolve_run(session_dir, scfg.get("run")))
        except FileNotFoundError:
            logger.warning("%s has no run; skipping", name)
            continue
        scfg = dict(scfg, run=outputs)
        (Path(outputs) / "plot_diagnostics.yaml").write_text(yaml.safe_dump(
            dict(config, sessions=[scfg]), sort_keys=False))
        script_copy = Path(outputs) / "plot_diagnostics_script.py"
        if Path(__file__).resolve() != script_copy.resolve():
            shutil.copyfile(__file__, script_copy)
        logger.info("=== %s", name)
        if not os.path.exists(os.path.join(outputs, "final_design_point.csv")):
            logger.warning("%s has no final_design_point.csv - not fitted; skipping",
                           name)
            continue
        plot_losses(session_dir, scfg, config,
                    os.path.join(outputs, config["loss_figure_name"]))
        plot_fit(session_dir, scfg, config,
                 os.path.join(outputs, config["fit_figure_name"]))

    logger.info("done")


if __name__ == "__main__":
    main()
