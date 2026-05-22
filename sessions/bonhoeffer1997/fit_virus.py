"""
Fit the 2-equation viral dynamics model from Bonhoeffer et al. 1997 (PNAS)
to virus load data.

Model (under RT-inhibitor drug therapy, beta -> 0):
    dy/dt = -a * y          (infected cells cleared at rate a)
    dv/dt =  k * y - u * v  (virus produced at rate k, cleared at rate u)

Only the product c = k * y(0) is identifiable from v(t) data alone,
so we fit 3 parameters: a, u, c.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data = np.array([
    [0.00000000, 100438.27445866732],
    [0.07960199,  95303.38546685888],
    [0.15920398, 154854.66302640623],
    [0.25373134, 147581.73641746133],
    [0.50248756, 119115.89410879342],
    [0.76616915, 118596.11761631008],
    [1.00995025, 112041.86124836495],
    [1.24875622, 112532.91210923201],
    [1.49751244,  75587.23447411254],
    [1.75124378,  84319.79225789376],
    [2.00497512,  72037.19343460193],
    [3.00497512,  38713.66575660163],
    [3.99502488,  24890.90950112656],
])
t_data = data[:, 0]
v_data = data[:, 1]
v0 = v_data[0]

# ---------------------------------------------------------------------------
# ODE
# ---------------------------------------------------------------------------
def ode(t, state, a, u, c):
    """
    state = [y_eff, v]
    y_eff = k * y  (so dy_eff/dt = -a * y_eff, and dv/dt = y_eff - u*v)
    c = initial value of y_eff = k * y(0)
    """
    y_eff, v = state
    return [-a * y_eff, y_eff - u * v]

def simulate(a, u, c, t_eval):
    sol = solve_ivp(
        ode,
        (t_eval[0], t_eval[-1]),
        [c, v0],
        args=(a, u, c),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
        dense_output=False,
    )
    if not sol.success:
        return None
    return sol.y[1]  # v(t)

# ---------------------------------------------------------------------------
# Loss (mean squared relative error in log-space — appropriate for log-scale data)
# ---------------------------------------------------------------------------
def loss(log_params):
    a, u, c = np.exp(log_params)
    v_pred = simulate(a, u, c, t_data)
    if v_pred is None or np.any(v_pred <= 0):
        return 1e10
    return np.mean((np.log(v_pred) - np.log(v_data)) ** 2)

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
# Initial guesses from paper:
#   a ~ ln(2)/2 days  (infected cell half-life ~2 days)
#   u ~ ln(2)/0.25 days  (free virus half-life ~6 hours)
#   c = k*y(0) ~ u*v0 at equilibrium (since at eq: k*y* = u*v*)
a_init = np.log(2) / 2.0      # ~0.35 /day
u_init = np.log(2) / 0.25     # ~2.77 /day
c_init = u_init * v0          # equilibrium assumption

x0 = np.log([a_init, u_init, c_init])

result = minimize(
    loss,
    x0,
    method="Nelder-Mead",
    options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-12},
)

a_fit, u_fit, c_fit = np.exp(result.x)

print("=" * 50)
print("Bonhoeffer 1997 — 2-equation model fit")
print("=" * 50)
print(f"  a = {a_fit:.4f} /day   (infected cell half-life = {np.log(2)/a_fit:.2f} days)")
print(f"  u = {u_fit:.4f} /day   (free virus half-life = {np.log(2)/u_fit*24:.1f} hours)")
print(f"  c = k·y(0) = {c_fit:.2f}")
print(f"  Final loss = {result.fun:.6f}")
print(f"  Optimizer success: {result.success}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
t_fine = np.linspace(t_data[0], t_data[-1], 500)
v_fit = simulate(a_fit, u_fit, c_fit, t_fine)

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(t_data, v_data, "o", color="black", label="Data", zorder=5)
ax.semilogy(t_fine, v_fit, "-", color="royalblue", linewidth=2, label="Model fit")
ax.set_xlabel("Time (days)", fontsize=12)
ax.set_ylabel("Plasma virus load", fontsize=12)
ax.set_title(
    f"Virus dynamics under drug therapy  (Bonhoeffer et al. 1997)\n"
    f"a = {a_fit:.3f}/day, u = {u_fit:.3f}/day",
    fontsize=11,
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("virus_fit.png", dpi=150)
print("Plot saved to virus_fit.png")
