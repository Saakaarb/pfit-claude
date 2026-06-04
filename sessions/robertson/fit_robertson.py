"""
Self-contained fit of the Robertson stiff ODE system.

State variables (3):
    y1, y2, y3  — species concentrations (conserved: y1+y2+y3 = 1)

ODEs:
    dy1/dt = -k1*y1 + k3*y2*y3
    dy2/dt =  k1*y1 - k2*y2^2 - k3*y2*y3
    dy3/dt =  k2*y2^2

Parameters to fit (3, all log-scale):
    k1  [0.001, 10]      true value: 0.04
    k2  [5e3,  5e10]     true value: 3e7
    k3  [1,    1e8]      true value: 1e4

Fixed: none
Initial conditions: y1=1, y2=0, y3=0

Loss: normalised RMSE over all three species (identical to user_model.py)

Equivalent budget to pfit-claude: 100 particles x 20 iters = 2000 evals
  -> scipy DE: popsize=34 (34x3=102~100), maxiter=20, workers=8

pfit-claude reference: k1=4.01e-2, k2=3.01e7, k3=1.00e4
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data   = np.loadtxt("../../examples/robertson_session/inputs/robertson_data.csv", delimiter=",")
t_data = data[:, 0]
y_data = data[:, 1:]   # shape (17, 3): y1, y2, y3

Y0 = np.array([1.0, 0.0, 0.0])

# Precompute scale factors (max per species, matching user_model.py)
_scale = np.max(y_data, axis=0)   # shape (3,)

# ---------------------------------------------------------------------------
# ODE
# ---------------------------------------------------------------------------
def ode(t, y, k1, k2, k3):
    y1, y2, y3 = y
    dy1 = -k1 * y1 + k3 * y2 * y3
    dy2 =  k1 * y1 - k2 * y2**2 - k3 * y2 * y3
    dy3 =  k2 * y2**2
    return [dy1, dy2, dy3]

# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
def simulate(k1, k2, k3, rtol=1e-3, atol=1e-3):
    try:
        sol = solve_ivp(
            ode,
            (t_data[0], t_data[-1]),
            Y0,
            args=(k1, k2, k3),
            t_eval=t_data,
            method="Radau",
            rtol=rtol,
            atol=atol,
        )
        if not sol.success or sol.y.shape[1] != len(t_data):
            return None
        return sol.y.T   # (17, 3)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Loss  (identical to _compute_loss_problem in user_model.py)
# ---------------------------------------------------------------------------
def loss(log_params, rtol=1e-3, atol=1e-3):
    k1 = 10.0 ** log_params[0]
    k2 = 10.0 ** log_params[1]
    k3 = 10.0 ** log_params[2]

    y_sim = simulate(k1, k2, k3, rtol=rtol, atol=atol)
    if y_sim is None:
        return 1e10

    return float(np.sqrt(np.mean(np.square((y_sim - y_data) / _scale))))

def loss_tight(log_params):
    return loss(log_params, rtol=1e-7, atol=1e-9)

# ---------------------------------------------------------------------------
# Global search — differential evolution
# ---------------------------------------------------------------------------
bounds = [
    (np.log10(0.001), np.log10(10.0)),   # log10(k1)
    (np.log10(5e3),   np.log10(5e10)),   # log10(k2)
    (np.log10(1.0),   np.log10(1e8)),    # log10(k3)
]

t_start = time.time()
print("Stage 1 — differential_evolution (102 individuals x 20 iters x 8 workers) ...")
de = differential_evolution(
    loss,
    bounds,
    maxiter=20,
    popsize=34,        # 34 x 3 params = 102 ~ pfit's 100 particles
    tol=1e-6,
    seed=42,
    workers=8,
    updating="deferred",
    disp=True,
)
t_de = time.time() - t_start
print(f"  DE finished: loss = {de.fun:.6f}  ({t_de:.2f} s)")

# ---------------------------------------------------------------------------
# Local polish — Nelder-Mead with tight tolerances
# ---------------------------------------------------------------------------
print("\nStage 2 — Nelder-Mead (tighter tolerances) ...")
t_nm0 = time.time()
nm = minimize(
    loss_tight,
    de.x,
    method="Nelder-Mead",
    options={"maxiter": 10000, "xatol": 1e-10, "fatol": 1e-12, "disp": True},
)
t_nm = time.time() - t_nm0
print(f"  NM finished: loss = {nm.fun:.6f}  ({t_nm:.2f} s)")

best = nm if nm.fun < de.fun else de
xb   = best.x
k1_fit = 10.0 ** xb[0]
k2_fit = 10.0 ** xb[1]
k3_fit = 10.0 ** xb[2]
t_total = time.time() - t_start

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
ref = dict(k1=0.04, k2=3e7, k3=1e4)
fit = dict(k1=k1_fit, k2=k2_fit, k3=k3_fit)

print("\n" + "=" * 60)
print(f"{'Param':>6}  {'Fitted':>14}  {'True value':>14}  {'ratio':>8}")
print("=" * 60)
for n in ["k1", "k2", "k3"]:
    r = ref[n]; f = fit[n]
    print(f"  {n:4s}  {f:14.4e}  {r:14.4e}  {f/r:8.4f}")
print(f"\n  Final loss      = {best.fun:.6f}")
print(f"  DE wall time    = {t_de:.2f} s")
print(f"  NM wall time    = {t_nm:.2f} s")
print(f"  Total wall time = {t_total:.2f} s")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
y_sim = simulate(k1_fit, k2_fit, k3_fit, rtol=1e-7, atol=1e-9)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle(
    f"Robertson System — Standalone Fit (DE + Nelder-Mead),  loss = {best.fun:.4f}\n"
    rf"$k_1={k1_fit:.3e}$,  $k_2={k2_fit:.3e}$,  $k_3={k3_fit:.3e}$",
    fontsize=10,
)

style_dat = dict(marker="o", ms=6, ls="none", color="k", label="Data", zorder=3)
style_sim = dict(lw=2, color="tab:red", label="Simulation")

for ax, col, yscale, name in zip(
    axes, [0, 1, 2], ["linear", "log", "linear"], ["$y_1$", "$y_2$", "$y_3$"]
):
    ax.semilogx(t_data, y_data[:, col], **style_dat)
    ax.semilogx(t_data, y_sim[:, col],  **style_sim)
    ax.set_yscale(yscale)
    ax.set_xlabel("Time")
    ax.set_ylabel(name + (" (log)" if yscale == "log" else ""))
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig("robertson_fit.png", dpi=150)
print("\nPlot saved to robertson_fit.png")
