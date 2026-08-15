"""
Post-fit sloppiness / identifiability diagnostics.

Computes the eigenvalue spectrum of the Hessian of the fitting loss at the
best-fit point, in log-parameter space -- the Fisher-information / "sloppiness"
spectrum in the sense of Gutenkunst et al. (2007) and Hass et al. (2019,
doi:10.1093/bioinformatics/btz020). A model fit is termed *sloppy* when the
eigenvalues span more than ~6 orders of magnitude, and any near-zero eigenvalue
marks a practically non-identifiable parameter combination.

This is intended to run automatically at the end of a gradient-based estimation
(see fit_equation_system / fit_gradient_only_system in helper_functions.py) and
is also callable stand-alone via analyze_fit.py.

Notes / caveats
  * It is the Hessian of the framework's (RMSE-type) loss, i.e. the Fisher
    information under that loss's implied noise model. Near the optimum the loss
    is a monotone function of the sum-of-squares, so the eigenVECTORS and the
    eigenvalue SPREAD (the sloppiness) match the Gauss-Newton FIM; only the
    absolute eigenvalue scale (hence absolute confidence intervals) depends on
    the assumed noise. Calibrated CIs would need the Gaussian FIM with fitted
    sigma and are not produced here.
  * LOCAL quadratic measure at the optimum; meaningful only at a converged fit.
  * Second-order autodiff through a stiff implicit solver is not always
    available, so the Hessian falls back to symmetric finite-differencing of the
    (first-order autodiff) gradient.
"""
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

SLOPPY_DECADES = 6.0        # sloppy if identifiable modes span > 6 orders of magnitude
NONIDENT_FLOOR = 1e-6       # |normalised eigenvalue| below this -> non-identifiable direction
MAX_NPAR = 60               # skip the (finite-difference) Hessian above this many parameters
FD_EPS = 1e-4               # finite-difference step in log-parameter space


def run_sloppiness_analysis(compute_loss_problem, constants_list, scaled_best_position,
                            param_names, output_dir):
    """Compute and write the sloppiness report + spectrum plot for a completed fit.

    Args:
        compute_loss_problem: the generated per-experiment loss fn (constants, scaled_theta)->scalar
        constants_list: list of per-experiment constants dicts (must already have
                        'min_limits', 'max_limits', 'is_logscale' set)
        scaled_best_position: best-fit parameters in the [-1, 1] search space
        param_names: list of trainable-parameter names (YAML order)
        output_dir: session outputs directory (report + plot are written here)

    Never raises: any failure is caught and reported, so it cannot break a fit.
    Returns a dict of the computed metrics (or None on skip/failure).
    """
    try:
        return _run(compute_loss_problem, constants_list, scaled_best_position,
                    param_names, Path(output_dir))
    except Exception as e:  # diagnostics must never crash the fit
        print(f"[sloppiness] skipped: {type(e).__name__}: {e}")
        return None


def _run(compute_loss_problem, constants_list, scaled_best_position, param_names, output_dir):
    pnames = list(param_names)
    npar = len(pnames)
    if npar > MAX_NPAR:
        print(f"[sloppiness] {npar} parameters (> {MAX_NPAR}); skipping Hessian diagnostic "
              f"(finite-difference cost too high).")
        return None

    c0 = constants_list[0]
    lo = np.asarray(c0["min_limits"], dtype=float)
    hi = np.asarray(c0["max_limits"], dtype=float)
    span = hi - lo
    scaled_star = np.asarray(scaled_best_position, dtype=float)
    # search coordinate u (= log10(theta) for logscale params); Hessian is taken w.r.t. u.
    u_star = lo + (scaled_star + 1.0) / 2.0 * span

    def loss_of_u(u):
        scaled = 2.0 * (u - lo) / span - 1.0
        losses = [compute_loss_problem(c, scaled) for c in constants_list]
        return jnp.mean(jnp.array(losses))

    u_j = jnp.asarray(u_star)
    loss_star = float(loss_of_u(u_j))
    gfun = jax.grad(loss_of_u)
    grad = np.asarray(gfun(u_j))
    print(f"[sloppiness] loss={loss_star:.4g}  |grad|_inf={np.abs(grad).max():.2g}  "
          f"computing {npar}x{npar} Hessian ...")

    # Hessian in log-parameter space: try 2nd-order AD, else finite-difference the gradient.
    try:
        H = np.asarray(jax.hessian(loss_of_u)(u_j))
        method = "autodiff"
    except Exception:
        H = np.zeros((npar, npar))
        for j in range(npar):
            du = np.zeros(npar); du[j] = FD_EPS
            gp = np.asarray(gfun(jnp.asarray(u_star + du)))
            gm = np.asarray(gfun(jnp.asarray(u_star - du)))
            H[:, j] = (gp - gm) / (2 * FD_EPS)
        method = "finite-difference"
    H = 0.5 * (H + H.T)

    evals, evecs = np.linalg.eigh(H)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    lam_max = evals[0]
    norm = evals / lam_max
    n_nonident = int(np.sum(np.abs(norm) < NONIDENT_FLOOR))
    ident = evals[(evals > 0) & (norm > NONIDENT_FLOOR)]
    spread = float(np.log10(lam_max / ident.min())) if ident.size and lam_max > 0 else float("inf")
    sloppy = (spread > SLOPPY_DECADES) or (n_nonident > 0)

    def top_loadings(vec, k=4):
        idx = np.argsort(np.abs(vec))[::-1][:k]
        return ", ".join(f"{pnames[i]}({vec[i]:+.2f})" for i in idx)

    # ---- report ----
    L = []
    L.append(f"Sloppiness / identifiability report")
    L.append("=" * 60)
    L.append(f"trainable parameters : {npar}  ({', '.join(pnames)})")
    L.append(f"experiments          : {len(constants_list)}")
    L.append(f"loss at best fit     : {loss_star:.6g}   (|grad|_inf = {np.abs(grad).max():.2g})")
    L.append(f"Hessian method       : {method}   (log10-parameter space)")
    L.append("")
    L.append("Eigenvalue spectrum of the loss Hessian (Fisher / sloppiness spectrum):")
    for i, (ev, nv) in enumerate(zip(evals, norm)):
        L.append(f"  lambda_{i+1:<2d} = {ev:+.4e}   (normalised {nv:+.2e})")
    L.append("")
    L.append(f"spread over identifiable modes : {spread:.2f} orders of magnitude")
    L.append(f"non-identifiable (flat) directions (normalised < {NONIDENT_FLOOR:g}) : {n_nonident}"
             + ("  <- effectively lambda=0 (spectrum spans >6 decades)" if n_nonident else ""))
    verdict = "SLOPPY / NON-IDENTIFIABLE" if sloppy else "well-determined (not sloppy)"
    if n_nonident and spread <= SLOPPY_DECADES:
        verdict += f"  ({n_nonident} flat direction(s) despite {spread:.1f}-decade identifiable spread)"
    L.append(f"VERDICT              : {verdict}")
    L.append("")
    L.append("STIFFEST direction (best-constrained combination):")
    L.append(f"   {top_loadings(evecs[:, 0])}")
    L.append("SLOPPIEST direction (least-constrained combination):")
    L.append(f"   {top_loadings(evecs[:, -1])}")
    L.append("")
    L.append("Note: the stiffest/sloppiest directions are combinations of parameters")
    L.append("(eigenvectors); they are NOT a per-parameter identifiability verdict.")
    L.append("")
    L.append("Caveats: local quadratic measure at the optimum; Hessian of the framework")
    L.append("RMSE loss (spread & eigenvectors match the Gauss-Newton FIM; absolute scale")
    L.append("is noise-model dependent); valid only at a converged fit.")
    report = "\n".join(L)

    out = Path(output_dir)
    (out / "sloppiness_report.txt").write_text(report + "\n")
    print("[sloppiness] " + verdict + f"  (spread {spread:.1f} dec, {n_nonident} flat)")

    # ---- spectrum plot (Hass et al. Fig-5 style) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 12, "font.weight": "bold", "axes.labelweight": "bold",
                             "axes.linewidth": 1.4, "savefig.bbox": "tight", "savefig.dpi": 200})
        fig, ax = plt.subplots(figsize=(4.6, 5.2))
        y = np.clip(np.abs(norm), 1e-20, None)
        for yi in y:
            ax.plot([-0.35, 0.35], [yi, yi], color="C0", lw=1.6, zorder=3)
        ax.scatter(np.zeros_like(y), y, s=70, color="C0", zorder=4)
        ax.axhspan(NONIDENT_FLOOR, 1.0, color="0.85", alpha=0.6, zorder=0, label="within 6 decades")
        ax.set_yscale("log"); ax.set_xlim(-1, 1); ax.set_xticks([])
        ax.set_ylabel("normalised eigenvalue  $\\lambda / \\lambda_{max}$")
        name = Path(output_dir).parent.name
        ax.set_title(f"{name}\nspread {spread:.1f} dec, {n_nonident} flat "
                     f"({'sloppy' if sloppy else 'not sloppy'})")
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "sloppiness_spectrum.png")
        plt.close(fig)
    except Exception as e:
        print(f"[sloppiness] plot skipped: {e}")

    return {"eigenvalues": evals, "spread": spread, "n_nonident": n_nonident,
            "sloppy": sloppy}
