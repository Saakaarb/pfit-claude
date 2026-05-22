"""
Self-contained fit of the sliding basepoint 2-mass friction model.

Physical system: two masses connected by an evolving spring + dashpot;
mass 2 has Coulomb friction with velocity threshold.

State variables (6):
    x1, x2  — positions of mass 1 and mass 2 (m)
    v1, v2  — velocities  (m/s)
    k        — evolving spring stiffness (N/m),  starts at 1.29e6
    c1       — evolving viscous damping  (N·s/m), starts at 0

Parameters to fit (5):
    c2  — Coulomb friction force limit    [1e4, 1e7],  log-scale
    Dk  — stiffness growth rate           [0.001, 10], log-scale
    Dc  — damping growth rate             [0.1, 1.0],  linear
    m1  — mass 1                          [100, 10000], log-scale
    m2  — mass 2                          [1, 100],    log-scale

Fixed: vf = 0.1  (velocity threshold for Coulomb friction)

Data columns: time (s) | force (kN) | displacement (m)

pfit-claude reference answer:
    c2=7.411e4, Dk=1.225e-2, Dc=5.988e-1, m1=2.003e3, m2=2.911e1
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
data = np.loadtxt("force_disp_data_sliding_basepoint.csv", delimiter=",")
t_data = data[:, 0]
F_data_N = data[:, 1] * 1000.0   # kN → N
s_data   = data[:, 2]            # displacement in m

# Precompute scale factors (used in loss)
_scale_F = float(np.max(np.log10(np.abs(F_data_N))))
_scale_s = float(np.max(np.abs(s_data)))

# ---------------------------------------------------------------------------
# Fixed parameters & initial conditions
# ---------------------------------------------------------------------------
VF = 0.1   # velocity threshold (m/s)
Y0 = np.array([0.0, 0.0, 11.06, 0.0, 1.29e6, 0.0])   # x1,x2,v1,v2,k,c1

# ---------------------------------------------------------------------------
# ODE
# ---------------------------------------------------------------------------
def ode(t, y, c2, Dk, Dc, m1, m2):
    x1, x2, v1, v2, k, c1 = y

    Fs = k * (x2 - x1)

    F1_val = Fs - c1 * np.abs(v1) * np.sign(v1)

    neg_Fs = -Fs
    if np.abs(neg_Fs) < c2 and np.abs(v2) < VF:
        F2_val = 0.0
    else:
        F2_val = neg_Fs - c2 * np.sign(v2)

    dv1dt = F1_val / m1
    dv2dt = F2_val / m2
    P = np.abs(m1 * v1 * dv1dt)

    return [v1, v2, dv1dt, dv2dt, Dk * P, Dc * P]

# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
def simulate(c2, Dk, Dc, m1, m2):
    try:
        sol = solve_ivp(
            ode,
            (t_data[0], t_data[-1]),
            Y0,
            args=(c2, Dk, Dc, m1, m2),
            t_eval=t_data,
            method="LSODA",
            rtol=1e-4,
            atol=[1e-6, 1e-6, 1e-3, 1e-3, 1e1, 1e-6],
        )
        if not sol.success or sol.y.shape[1] != len(t_data):
            return None
        return sol
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Loss  (identical formula to user_model.py)
# ---------------------------------------------------------------------------
def loss(log_params):
    c2 = 10.0 ** log_params[0]
    Dk = 10.0 ** log_params[1]
    Dc =        log_params[2]    # linear scale
    m1 = 10.0 ** log_params[3]
    m2 = 10.0 ** log_params[4]

    sol = simulate(c2, Dk, Dc, m1, m2)
    if sol is None:
        return 1e10

    x1 = sol.y[0]; x2 = sol.y[1]
    v1 = sol.y[2]; k  = sol.y[4]; c1 = sol.y[5]

    Fs    = k * (x2 - x1)
    F_sim = np.abs(Fs - c1 * np.abs(v1) * np.sign(v1))
    s_sim = x1

    if np.any(F_sim[1:] <= 0) or np.any(~np.isfinite(F_sim)):
        return 1e10

    F_loss = np.sqrt(np.mean(((np.log10(F_sim[1:]) - np.log10(F_data_N[1:])) / _scale_F) ** 2))
    s_loss = np.sqrt(np.mean(((s_data - s_sim) / _scale_s) ** 2))
    return 10.0 * F_loss + s_loss

# ---------------------------------------------------------------------------
# Global search — differential evolution (analogous to PSO in pfit-claude)
# ---------------------------------------------------------------------------
# Bounds in the same space as loss(): log10 for log-scale, linear for Dc
bounds = [
    (np.log10(1e4), np.log10(1e7)),    # log10(c2)
    (np.log10(0.001), np.log10(10)),   # log10(Dk)
    (0.1, 1.0),                         # Dc  (linear)
    (np.log10(100), np.log10(10000)),  # log10(m1)
    (np.log10(1), np.log10(100)),      # log10(m2)
]

t_start = time.time()
print("Stage 1 — differential_evolution (global search) ...")
de = differential_evolution(
    loss,
    bounds,
    maxiter=20,
    popsize=100,           # 100 × 5 params = 500 individuals, matching pfit NUM_PARTICLES=500
    tol=1e-5,
    seed=42,
    workers=8,             # matching pfit PROCESSORS=8
    updating="deferred",   # required when workers > 1
    disp=True,
)
print(f"  DE finished: loss = {de.fun:.6f}")

# ---------------------------------------------------------------------------
# Local polish — Nelder-Mead
# ---------------------------------------------------------------------------
print("\nStage 2 — Nelder-Mead (local polish) ...")
nm = minimize(
    loss,
    de.x,
    method="Nelder-Mead",
    options={"maxiter": 10000, "xatol": 1e-9, "fatol": 1e-11, "disp": True},
)
print(f"  NM finished: loss = {nm.fun:.6f}")

best = nm if nm.fun < de.fun else de
xb = best.x
c2_fit = 10.0 ** xb[0]
Dk_fit = 10.0 ** xb[1]
Dc_fit =        xb[2]
m1_fit = 10.0 ** xb[3]
m2_fit = 10.0 ** xb[4]

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
ref = dict(c2=7.411e4, Dk=1.225e-2, Dc=5.988e-1, m1=2.003e3, m2=2.911e1)
fit = dict(c2=c2_fit,  Dk=Dk_fit,   Dc=Dc_fit,   m1=m1_fit,  m2=m2_fit)

print("\n" + "=" * 60)
print(f"{'Param':>6}  {'Fitted':>14}  {'pfit reference':>14}  {'ratio':>8}")
print("=" * 60)
for n in ["c2", "Dk", "Dc", "m1", "m2"]:
    r = ref[n]; f = fit[n]
    print(f"  {n:4s}  {f:14.4e}  {r:14.4e}  {f/r:8.3f}")
print(f"\n  Final loss = {best.fun:.6f}")
print(f"  Total wall time = {time.time() - t_start:.1f} s")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
sol = simulate(c2_fit, Dk_fit, Dc_fit, m1_fit, m2_fit)

x1 = sol.y[0]; x2 = sol.y[1]
v1 = sol.y[2]; k  = sol.y[4]; c1 = sol.y[5]
Fs    = k * (x2 - x1)
F_sim = np.abs(Fs - c1 * np.abs(v1) * np.sign(v1))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Sliding Basepoint — Standalone Fit (DE + Nelder-Mead)", fontsize=12)

axes[0].semilogy(t_data, F_data_N, "k.", ms=4, label="Experiment")
axes[0].semilogy(t_data, F_sim,    "r-", lw=2, label="Simulation")
axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Force F (N)")
axes[0].set_ylim(1e4, 4e5)
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(t_data, s_data, "k.", ms=4, label="Experiment")
axes[1].plot(t_data, x1,    "r-", lw=2, label="Simulation")
axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Displacement x1 (m)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("sliding_basepoint_fit.png", dpi=150)
print("Plot saved to sliding_basepoint_fit.png")
