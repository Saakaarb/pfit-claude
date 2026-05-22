"""
Self-contained fit of the ARC (Accelerating Rate Calorimetry) model
to NMC battery thermal runaway data.

ODE system (3 state variables):
    dc1/dt = -A1 * exp(-Ea1/(kb*T)) * c1
    dc2/dt =  A2 * exp(-Ea2/(kb*T)) * c2^n2 * (1-c2)^m2
    dT/dt  = |h1*dc1/dt| + |h2*dc2/dt|

Trainable parameters (8):
    Ea1, Ea2  activation energies   [2e-19, 4e-19] J,       log-scale
    h1,  h2   heat coefficients     [10, 1000] K,            log-scale
    A1,  A2   pre-exponential       [1e8, 1e25],             log-scale
    m2,  n2   reaction exponents    [1, 5],                  linear

Fixed: kb = 1.38e-23 J/K
Initial conditions: c1=1.0, c2=0.04, T=354 K

Matching pfit-claude budget: 350 particles x 50 iters = 17,500 evals
  -> scipy DE: popsize=44 (44x8=352~350), maxiter=50, workers=8
     (PSO tolerances: rtol=1e-3, atol=1e-3 — matches PSO_STEPSIZE_*)
  -> Nelder-Mead polish (tighter tolerances)

pfit-claude reference answer:
    Ea1=2.607e-19, h1=103.7, A1=6.200e13, A2=5.778e17,
    Ea2=2.538e-19, h2=237.5, m2=4.281,    n2=1.116
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
data    = np.loadtxt("NMC_SOC80_M2_normalized_converted_units.csv", delimiter=",")
t_data  = data[:, 0]   # time (s)
T_data  = data[:, 1]   # temperature (K)
hr_data = data[:, 2]   # heat rate (K/s)

# ---------------------------------------------------------------------------
# Fixed / initial conditions
# ---------------------------------------------------------------------------
KB = 1.38e-23
Y0 = np.array([1.0, 0.04, 354.0])   # c1, c2, T

# ---------------------------------------------------------------------------
# ODE
# ---------------------------------------------------------------------------
def rhs(Ea1, h1, A1, A2, Ea2, h2, m2, n2, c1, c2, T):
    dc1dt = -A1 * np.exp(-Ea1 / (KB * T)) * c1
    dc2dt =  A2 * np.exp(-Ea2 / (KB * T)) * (c2 ** n2) * ((1.0 - c2) ** m2)
    dTdt  = np.abs(h1 * dc1dt) + np.abs(h2 * dc2dt)
    return dc1dt, dc2dt, dTdt

def ode(t, y, Ea1, h1, A1, A2, Ea2, h2, m2, n2):
    c1, c2, T = y
    dc1dt, dc2dt, dTdt = rhs(Ea1, h1, A1, A2, Ea2, h2, m2, n2, c1, c2, T)
    return [dc1dt, dc2dt, dTdt]

def simulate(params, rtol, atol):
    Ea1, h1, A1, A2, Ea2, h2, m2, n2 = params
    try:
        sol = solve_ivp(
            ode,
            (t_data[0], t_data[-1]),
            Y0,
            args=(Ea1, h1, A1, A2, Ea2, h2, m2, n2),
            t_eval=t_data,
            method="Radau",
            rtol=rtol,
            atol=atol,
            max_step=500.0,
        )
        if not sol.success or sol.y.shape[1] != len(t_data):
            return None
        return sol
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Loss  (identical formula to user_model.py)
# ---------------------------------------------------------------------------
_eps      = 1e-12
_scale_hr = np.log10(np.max(hr_data + _eps))
_scale_T  = np.max(np.abs(T_data))

def compute_loss(sol, params):
    Ea1, h1, A1, A2, Ea2, h2, m2, n2 = params
    c1_s = sol.y[0]; c2_s = sol.y[1]; T_s = sol.y[2]

    # heat rate = dT/dt evaluated on the solution (vectorised)
    dc1dt, dc2dt, _ = rhs(Ea1, h1, A1, A2, Ea2, h2, m2, n2, c1_s, c2_s, T_s)
    hr_pred = np.abs(h1 * dc1dt) + np.abs(h2 * dc2dt)

    if np.any(~np.isfinite(hr_pred)) or np.any(hr_pred + _eps <= 0):
        return 1e6

    loss1 = np.sqrt(np.mean(((np.log10(hr_pred + _eps) - np.log10(hr_data + _eps)) / _scale_hr) ** 2))
    loss2 = 10.0 * np.mean(np.abs((T_data - T_s) / _scale_T))

    c1_f = c1_s[-1]; c2_f = c2_s[-1]; T_f = T_s[-1]
    loss3 = 500.0 if (c1_f > 0.02 or c2_f < 0.98 or abs(T_f - T_data[-1]) > 50) else 0.0

    return loss1 + loss2 + loss3

# DE phase uses loose tolerances (matching PSO_STEPSIZE_RTOL/ATOL = 1e-3)
def loss_de(log_params):
    Ea1 = 10.0 ** log_params[0]; h1  = 10.0 ** log_params[1]
    A1  = 10.0 ** log_params[2]; A2  = 10.0 ** log_params[3]
    Ea2 = 10.0 ** log_params[4]; h2  = 10.0 ** log_params[5]
    m2  = log_params[6];         n2  = log_params[7]
    params = [Ea1, h1, A1, A2, Ea2, h2, m2, n2]
    sol = simulate(params, rtol=1e-3, atol=1e-3)
    return 1e6 if sol is None else compute_loss(sol, params)

# NM phase uses tight tolerances (matching STEPSIZE_RTOL/ATOL = 1e-7/1e-9)
def loss_nm(log_params):
    Ea1 = 10.0 ** log_params[0]; h1  = 10.0 ** log_params[1]
    A1  = 10.0 ** log_params[2]; A2  = 10.0 ** log_params[3]
    Ea2 = 10.0 ** log_params[4]; h2  = 10.0 ** log_params[5]
    m2  = log_params[6];         n2  = log_params[7]
    params = [Ea1, h1, A1, A2, Ea2, h2, m2, n2]
    sol = simulate(params, rtol=1e-7, atol=1e-9)
    return 1e6 if sol is None else compute_loss(sol, params)

# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
bounds = [
    (np.log10(2e-19), np.log10(4e-19)),   # log10(Ea1)
    (np.log10(10),    np.log10(1000)),     # log10(h1)
    (np.log10(1e8),   np.log10(1e25)),     # log10(A1)
    (np.log10(1e8),   np.log10(1e25)),     # log10(A2)
    (np.log10(2e-19), np.log10(4e-19)),   # log10(Ea2)
    (np.log10(10),    np.log10(1000)),     # log10(h2)
    (1.0, 5.0),                            # m2 (linear)
    (1.0, 5.0),                            # n2 (linear)
]

t_start = time.time()
print("Stage 1 — differential_evolution (352 particles x 50 iters x 8 workers) ...")
de = differential_evolution(
    loss_de,
    bounds,
    maxiter=50,
    popsize=44,        # 44 x 8 params = 352 ~ pfit's 350 particles
    tol=1e-5,
    seed=42,
    workers=8,
    updating="deferred",
    disp=True,
)
t_de = time.time() - t_start
print(f"  DE finished: loss = {de.fun:.4f}  ({t_de:.1f} s)")

print("\nStage 2 — Nelder-Mead (local polish, tighter tolerances) ...")
t_nm0 = time.time()
nm = minimize(
    loss_nm,
    de.x,
    method="Nelder-Mead",
    options={"maxiter": 10000, "xatol": 1e-9, "fatol": 1e-11, "disp": True},
)
t_nm = time.time() - t_nm0
print(f"  NM finished: loss = {nm.fun:.4f}  ({t_nm:.1f} s)")

best = nm if nm.fun < de.fun else de
xb   = best.x
params_fit = [
    10.0**xb[0], 10.0**xb[1], 10.0**xb[2], 10.0**xb[3],
    10.0**xb[4], 10.0**xb[5], xb[6], xb[7],
]
Ea1_f, h1_f, A1_f, A2_f, Ea2_f, h2_f, m2_f, n2_f = params_fit
t_total = time.time() - t_start

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
ref = dict(Ea1=2.607e-19, h1=103.7,  A1=6.200e13, A2=5.778e17,
           Ea2=2.538e-19, h2=237.5,  m2=4.281,    n2=1.116)
fit = dict(Ea1=Ea1_f,     h1=h1_f,   A1=A1_f,     A2=A2_f,
           Ea2=Ea2_f,     h2=h2_f,   m2=m2_f,     n2=n2_f)

print("\n" + "=" * 65)
print(f"{'Param':>6}  {'Fitted':>14}  {'pfit reference':>14}  {'ratio':>8}")
print("=" * 65)
for n in ["Ea1", "h1", "A1", "A2", "Ea2", "h2", "m2", "n2"]:
    r = ref[n]; f = fit[n]
    print(f"  {n:4s}  {f:14.4e}  {r:14.4e}  {f/r:8.3f}")
print(f"\n  Final loss      = {best.fun:.4f}")
print(f"  DE wall time    = {t_de:.1f} s  ({t_de/60:.1f} min)")
print(f"  NM wall time    = {t_nm:.1f} s")
print(f"  Total wall time = {t_total:.1f} s  ({t_total/60:.1f} min)")

# ---------------------------------------------------------------------------
# Plot (same layout as examples/ARC_fitting/outputs/fit_result.png)
# ---------------------------------------------------------------------------
sol = simulate(params_fit, rtol=1e-7, atol=1e-9)
T_sim   = sol.y[2]
c1s = sol.y[0]; c2s = sol.y[1]
_, _, _ = rhs(*params_fit, c1s, c2s, T_sim)
dc1dt_s, dc2dt_s, _ = rhs(Ea1_f, h1_f, A1_f, A2_f, Ea2_f, h2_f, m2_f, n2_f,
                            c1s, c2s, T_sim)
hr_pred = np.abs(h1_f * dc1dt_s) + np.abs(h2_f * dc2dt_s)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"ARC Fitting — Standalone (DE + Nelder-Mead), loss={best.fun:.3f}", fontsize=12)

axes[0].plot(t_data, T_data, "k.", ms=3, label="Experiment")
axes[0].plot(t_data, T_sim,  "r-", lw=2, label="Simulation")
axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Temperature (K)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].semilogy(T_data, hr_data,  "k.", ms=3, label="Experiment")
axes[1].semilogy(T_sim,  hr_pred,  "r-", lw=2, label="Simulation")
axes[1].set_xlabel("Temperature (K)"); axes[1].set_ylabel("Heat rate dT/dt (K/s)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("arc_fit.png", dpi=150)
print("Plot saved to arc_fit.png")
