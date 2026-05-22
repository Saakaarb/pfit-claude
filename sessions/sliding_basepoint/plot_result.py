import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.loadtxt("force_disp_data_sliding_basepoint.csv", delimiter=",")
t_data = data[:, 0]
F_data_N = data[:, 1] * 1000.0
s_data   = data[:, 2]

VF = 0.1
Y0 = np.array([0.0, 0.0, 11.06, 0.0, 1.29e6, 0.0])

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

# Best parameters from the fit
c2, Dk, Dc, m1, m2 = 3.9287e+04, 1.1004e-03, 6.8849e-01, 2.2062e+03, 9.1492e+01

sol = solve_ivp(ode, (t_data[0], t_data[-1]), Y0, args=(c2, Dk, Dc, m1, m2),
                t_eval=t_data, method="LSODA", rtol=1e-4,
                atol=[1e-6, 1e-6, 1e-3, 1e-3, 1e1, 1e-6])

x1 = sol.y[0]; x2 = sol.y[1]; v1 = sol.y[2]; k = sol.y[4]; c1 = sol.y[5]
Fs    = k * (x2 - x1)
F_sim = np.abs(Fs - c1 * np.abs(v1) * np.sign(v1))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Sliding Basepoint — Standalone Fit (DE + Nelder-Mead)", fontsize=12)

axes[0].semilogy(t_data, F_data_N, "k.", ms=4, label="Experiment")
axes[0].semilogy(t_data, F_sim,    "r-", lw=2, label="Simulation")
axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Force F (N)")
axes[0].set_ylim(1e4, 4e5)   # match examples/sliding_basepoint/outputs/fit_result.png
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(t_data, s_data, "k.", ms=4, label="Experiment")
axes[1].plot(t_data, x1,    "r-", lw=2, label="Simulation")
axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Displacement x1 (m)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("sliding_basepoint_fit.png", dpi=150)
print("Done")
