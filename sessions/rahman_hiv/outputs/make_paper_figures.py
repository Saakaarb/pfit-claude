"""Generate paper-ready figures for the rahman_hiv session (single-experiment fit).

Produces (all written next to this script, existing figures untouched):
  * loss_history.png              - DE global-search and NODE gradient loss vs
                                    iteration, with the DE optimum carried onto the
                                    gradient panel as a reference point + line.
  * fit_prevalence.png            - the single observable (HIV prevalence) data vs model
  * sloppiness_spectrum_paper.png - publication-styled eigenvalue (sloppiness) spectrum

Inputs (already present in this outputs directory):
  de_fitting.log, NODE_fitting.log, result_solution_exp1.csv, sloppiness_report.txt

result_solution_exp1.csv has 3 columns: time (years), data prevalence (%),
model prevalence (%).

No CLI arguments: every path is derived from this script's own location.
"""
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
LOG_FILE = OUT / "make_paper_figures.log"

# ----------------------------------------------------------------------------- logging
logger = logging.getLogger("rahman_figures")
logger.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.DEBUG)
_sh.setFormatter(_fmt)
_fh = logging.FileHandler(LOG_FILE)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
logger.handlers[:] = [_sh, _fh]

# ----------------------------------------------------------------------------- style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "axes.linewidth": 1.1,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DATA_COLOR = "#222222"
FIT_COLOR = "#c0392b"
DE_COLOR = "#2c7fb8"


def _read_loss_log(path: Path):
    """Parse an 'iter, loss, time' loss log (skips header lines). Returns (iter, loss)."""
    iters, losses = [], []
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            iters.append(int(float(parts[0])))
            losses.append(float(parts[1]))
        except ValueError:
            continue  # header / decorative lines
    logger.debug("parsed %d points from %s", len(iters), path.name)
    return np.array(iters, dtype=float), np.array(losses, dtype=float)


def plot_loss_history():
    de_it, de_loss = _read_loss_log(OUT / "de_fitting.log")
    node_it, node_loss = _read_loss_log(OUT / "NODE_fitting.log")
    de_best = float(de_loss[-1])  # last (best) loss of the global search
    ylabel = "peak-normalised RMSE (prevalence)"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # (a) global search
    ax1.plot(de_it, de_loss, "-o", color=DE_COLOR, markersize=4, lw=1.8)
    ax1.set_xlabel("DE generation")
    ax1.set_ylabel(ylabel)
    ax1.set_title("(a) Global search (Differential Evolution)")
    ax1.grid(alpha=0.25)

    # (b) gradient refinement, with the DE optimum carried over as a reference
    ax2.axhline(de_best, ls="--", lw=1.3, color=DE_COLOR, alpha=0.75, zorder=1,
                label="DE optimum (reference)")
    ax2.plot(node_it, node_loss, "-", color=FIT_COLOR, lw=1.8, zorder=2,
             label="gradient refinement")
    ax2.plot([0], [de_best], marker="*", markersize=16, color=DE_COLOR,
             markeredgecolor="white", markeredgewidth=0.8, zorder=3, ls="none",
             label="DE hand-off")
    ax2.set_xlabel("gradient iteration")
    ax2.set_ylabel(ylabel)
    ax2.set_title("(b) Gradient refinement (NODE)")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    path = OUT / "loss_history.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("wrote %s  (DE best=%.4e, NODE final=%.4e)", path.name, de_best, node_loss[-1])


def plot_estimate():
    data = np.loadtxt(OUT / "result_solution_exp1.csv", delimiter=",")
    t, d, m = data[:, 0], data[:, 1], data[:, 2]
    scale = max(np.max(np.abs(d)), 1e-12)
    rmse = float(np.sqrt(np.mean(((m - d) / scale) ** 2)))

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(t, d, "o", color=DATA_COLOR, markersize=7, label="data (Rahman et al. 2016)",
            markerfacecolor="white", markeredgewidth=1.6)
    ax.plot(t, m, "-", color=FIT_COLOR, lw=2.2, label="pfit model")
    ax.set_xlabel("time (years)")
    ax.set_ylabel("HIV prevalence (%)")
    ax.set_title(f"HIV prevalence   (peak-norm. RMSE = {rmse:.3f})")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUT / "fit_prevalence.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("wrote %s  (RMSE=%.4f)", path.name, rmse)


def _parse_sloppiness_report():
    """Extract normalised eigenvalues + summary numbers from sloppiness_report.txt."""
    text = (OUT / "sloppiness_report.txt").read_text()
    norm = [float(m) for m in re.findall(r"normalised\s*([+-]?\d[\d.eE+-]*)", text)]
    spread = re.search(r"spread over identifiable modes\s*:\s*([\d.]+)", text)
    nflat = re.search(r"non-identifiable \(flat\) directions.*?:\s*(\d+)", text)
    verdict = re.search(r"VERDICT\s*:\s*(.+)", text)
    return (np.array(norm),
            float(spread.group(1)) if spread else float("nan"),
            int(nflat.group(1)) if nflat else 0,
            verdict.group(1).strip() if verdict else "")


def plot_sloppiness_paper():
    norm, spread, nflat, verdict = _parse_sloppiness_report()
    if norm.size == 0:
        logger.warning("no eigenvalues parsed from sloppiness_report.txt; skipping")
        return
    y = np.clip(np.abs(norm), 1e-20, None)
    NONIDENT_FLOOR = 1e-6

    fig, ax = plt.subplots(figsize=(4.4, 5.6))
    ax.axhspan(NONIDENT_FLOOR, 1.0, color="#dfe9f3", alpha=0.9, zorder=0,
               label="6-decade identifiable band")
    ax.axhline(NONIDENT_FLOOR, color="#8fa8c8", lw=1.0, ls="--", zorder=1)

    for yi in y:
        color = FIT_COLOR if yi < NONIDENT_FLOOR else "#1f4e79"
        ax.plot([-0.4, 0.4], [yi, yi], color=color, lw=2.2, zorder=3)
    colors = [FIT_COLOR if yi < NONIDENT_FLOOR else "#1f4e79" for yi in y]
    ax.scatter(np.zeros_like(y), y, s=90, c=colors, zorder=4, edgecolors="white", linewidths=0.8)

    ax.set_yscale("log")
    ax.set_xlim(-1, 1)
    ax.set_xticks([])
    ax.set_ylabel(r"normalised eigenvalue  $\lambda_i / \lambda_{\max}$")
    is_sloppy = "SLOPPY" in verdict.upper()
    ax.set_title(f"rahman_hiv sloppiness spectrum\n"
                 f"{spread:.1f}-decade identifiable spread, {nflat} flat "
                 f"({'sloppy' if is_sloppy else 'not sloppy'})",
                 fontsize=12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), fontsize=9)
    fig.tight_layout()
    path = OUT / "sloppiness_spectrum_paper.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("wrote %s  (spread=%.1f dec, %d flat)", path.name, spread, nflat)


def main():
    logger.info("generating paper figures for rahman_hiv in %s", OUT)
    plot_loss_history()
    plot_estimate()
    plot_sloppiness_paper()
    logger.info("done")


if __name__ == "__main__":
    main()
