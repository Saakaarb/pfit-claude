"""Benchmark stiffness estimators over a parameter box against known answers.

For each test system we sample (x, p) from X x P, form J = df/dy by autodiff,
and compare:

  truth      max_i |Re lambda_i(J)|  (direct eigendecomposition)
  bendixson  lambda_max(J_sym) / lambda_min(J_sym)   [matrix measure, mu_2]
  gershgorin |J_ii| + sum_{j!=i} |J_ij|              [the current R1 method]
  rohn       lambda_max(Jc) + rho(Delta) on the interval matrix
  hertz      exact extreme eigenvalue over the symmetric interval matrix
  ratecons   max/min of the rate constants appearing in the box [the shortcut]

Decision-relevant scalar: N_explicit = T_span * max|Re lambda|, the number of
steps an explicit method needs for stability. Large => stiff.
"""
import itertools
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
kb = 1.380649e-23


# --------------------------------------------------------------------------
# test systems: (name, rhs(y,p), state box, param box (log flags), T_span,
#                verdict we already believe, rate-constant range over the box)
# --------------------------------------------------------------------------

def decay(y, p):
    k1, k2 = p
    return jnp.array([-k1 * y[0], k1 * y[0] - k2 * y[1]])


def robertson(y, p):
    k1, k2, k3 = p
    y1, y2, y3 = y
    return jnp.array([-k1 * y1 + k3 * y2 * y3,
                      k1 * y1 - k3 * y2 * y3 - k2 * y2**2,
                      k2 * y2**2])


def rahman(y, p):
    rel_n, beta_m, rel_w, tau_w, w_n, w_m, g_m, g_w, bcr = p
    S, In, Im, Iw, Tn, Tm, Tw = y
    bn, bw, bt = rel_n * beta_m, rel_w * beta_m, 0.04 * beta_m
    N = S + In + Im + Iw + Tn + Tm + Tw
    tot = In + Im + Iw + Tn + Tm + Tw
    lam = ((bn * In + beta_m * Im + bw * Iw + bt * (Tn + Tm + Tw)) / N) * jnp.exp(-bcr * tot)
    mu = dict(S=0.0288, In=0.0888, Im=0.1368, Iw=0.3108, Tn=0.0408, Tm=0.0528, Tw=0.1752)
    return jnp.array([
        1032672.0 - lam * S - mu['S'] * S,
        lam * S - w_n * In - mu['In'] * In,
        w_n * In - w_m * Im - mu['Im'] * Im,
        w_m * Im - tau_w * Iw - mu['Iw'] * Iw,
        g_m * Tm - mu['Tn'] * Tn,
        g_w * Tw - g_m * Tm - mu['Tm'] * Tm,
        tau_w * Iw - g_w * Tw - mu['Tw'] * Tw,
    ])


def arc(y, p):
    Ea1, A1, n1, h1, Ea2, A2, m2, h2 = p
    c1, c2, T = y
    d1 = -A1 * jnp.exp(-Ea1 / (kb * T)) * jnp.abs(c1)**n1
    d2 = A2 * jnp.exp(-Ea2 / (kb * T)) * jnp.abs(1.0 - c2)**m2
    # smooth surrogate for the gate so the Jacobian exists; the switch itself is
    # the smoothness axis, not the stiffness axis
    gate = 0.5 * (1 + jnp.tanh((T - 485.0) / 1.0))
    return jnp.array([d1, d2, jnp.abs(h1 * d1) + gate * jnp.abs(h2 * d2)])


def sneyd(y, p):
    k1, k2, k3, k4, k_1, k_2, k_3, k_4, l2, l4, l6, l_2, l_4, l_6 = p
    O, R, I1, S, A, I2, IP3, Ca = y
    L1 = k_1 * l2 / (k1 * l_2); L3 = k_2 * l4 / (k2 * l_4); L5 = k_4 * l6 / (k4 * l_6)
    v0 = (k_2 + l_4 * Ca) / (1.0 + Ca / L5) * O
    v1 = (k2 * L3 + l4 * Ca) / (L3 + Ca * (1.0 + L3 / L1)) * IP3 * R
    v2 = (k1 * L1 + l2) * Ca / (L1 + Ca * (1.0 + L1 / L3)) * R
    v3 = (k_1 + l_2) * I1
    v4 = (k4 * L5 + l6) * Ca / (L5 + Ca) * O
    v5 = L1 * (k_4 + l_6) / (L1 + Ca) * A
    v6 = (k1 * L1 + l2) * Ca / (L1 + Ca) * A
    v7 = (k_1 + l_2) * I2
    v8 = k3 * L5 / (L5 + Ca) * O
    v9 = k_3 * S
    return jnp.array([-v0 + v1 - v4 + v5 - v8 + v9, v0 - v1 - v2 + v3, v2 - v3,
                      v8 - v9, v4 - v5 - v6 + v7, v6 - v7, 0.0, 0.0])


SYSTEMS = [
    dict(name="decay chain", f=decay, believed="NOT stiff",
         xbox=[(0, 1), (0, 1)],
         pbox=[(0.1, 10, 1), (0.03, 3, 1)], T=10.0, rates=(0.03, 10)),
    dict(name="robertson", f=robertson, believed="STIFF (textbook)",
         xbox=[(0, 1), (0, 1e-4), (0, 1)],
         pbox=[(1e-3, 10, 1), (5e3, 5e10, 1), (1, 1e8, 1)], T=1e5, rates=(1e-3, 5e10)),
    dict(name="rahman HIV", f=rahman, believed="NOT stiff (claimed)",
         xbox=[(1e6, 1.8e7), (0, 1e6), (0, 1e6), (0, 1e6), (0, 1e6), (0, 1e6), (0, 1e6)],
         pbox=[(0.1, 1000, 1), (1e-4, 1, 1), (0.1, 1000, 1), (1e-4, 1, 1), (1e-4, 1, 1),
               (1e-4, 1, 1), (1e-4, 1, 1), (1e-4, 1, 1), (1e-9, 1e-5, 1)],
         T=22.0, rates=(1e-4, 1.0)),
    dict(name="sneyd IPR", f=sneyd, believed="STIFF (claimed)",
         xbox=[(0, 1)] * 6 + [(3, 10), (0.1, 10)],
         pbox=[(1e-3, 1e5, 1)] * 14, T=1.0, rates=(1e-3, 1e5)),
    dict(name="ARC", f=arc, believed="CONTESTED",
         xbox=[(0, 1), (0, 1), (354, 700)],
         pbox=[(1e-19, 3e-19, 1), (1e5, 1e25, 1), (0.5, 3, 0), (1, 1e3, 1),
               (2e-19, 4e-19, 1), (1e8, 1e25, 1), (1, 8, 0), (1, 1.2e3, 1)],
         T=5.59e4, rates=None),
]


def sample(bounds, rng, n):
    out = []
    for b in bounds:
        if len(b) == 3 and b[2] == 1:
            out.append(10**rng.uniform(np.log10(b[0]), np.log10(b[1]), n))
        else:
            out.append(rng.uniform(b[0], b[1], n))
    return np.stack(out, axis=1)


def _hertz_lambda_max(Jc, Delta):
    """Hertz: max eigenvalue over a symmetric interval matrix, at a vertex.

    (A_z)_ij = c_ij + z_i z_j r_ij. Note z_i^2 = 1, so the DIAGONAL is pinned at
    c_ii + r_ii — its maximum. That makes this correct for lambda_max only.
    """
    n = Jc.shape[0]
    hi = -np.inf
    for z in itertools.product([1, -1], repeat=n - 1):
        z = np.array((1,) + z, dtype=float)
        hi = max(hi, np.linalg.eigvalsh(Jc + np.outer(z, z) * Delta).max())
    return hi


def hertz_max(Jc, Delta):
    """Both extremes. lambda_min([A]) = -lambda_max([-A]), which flips the
    diagonal to its minimum -- doing this by reading lambda_min off the same
    vertex set silently UNDERestimates, the dangerous direction."""
    hi = _hertz_lambda_max(Jc, Delta)
    lo = -_hertz_lambda_max(-Jc, Delta)
    return hi, lo


def run(sys, n_samples=4000, seed=0):
    rng = np.random.default_rng(seed)
    f = sys["f"]
    jac = jax.jit(jax.jacfwd(lambda y, p: f(y, p), argnums=0))
    X = sample(sys["xbox"], rng, n_samples)
    P = sample(sys["pbox"], rng, n_samples)

    Js = []
    true_absc, true_ratio, bend_hi, bend_lo, gersh = [], [], [], [], []
    for x, p in zip(X, P):
        J = np.asarray(jac(jnp.array(x), jnp.array(p)))
        if not np.all(np.isfinite(J)):
            continue
        Js.append(J)
        w = np.linalg.eigvals(J)
        re = np.abs(w.real)
        true_absc.append(re.max())
        nz = re[re > 1e-300]
        true_ratio.append(re.max() / nz.min() if nz.size else np.inf)
        Jsym = 0.5 * (J + J.T)
        ev = np.linalg.eigvalsh(Jsym)
        bend_hi.append(ev.max()); bend_lo.append(ev.min())
        gersh.append(np.max(np.abs(np.diag(J)) + (np.abs(J).sum(1) - np.abs(np.diag(J)))))

    Js = np.array(Js)
    if Js.size == 0:
        return None
    # empirical interval enclosure of J over the box
    Jlo, Jhi = Js.min(0), Js.max(0)
    Jc = 0.5 * (Jlo + Jhi); R = 0.5 * (Jhi - Jlo)
    Jc_s = 0.5 * (Jc + Jc.T); R_s = 0.5 * (R + R.T)
    rohn_hi = np.linalg.eigvalsh(Jc_s).max() + np.abs(np.linalg.eigvals(R_s)).max()
    rohn_lo = np.linalg.eigvalsh(Jc_s).min() - np.abs(np.linalg.eigvals(R_s)).max()
    n = Jc_s.shape[0]
    hz = hertz_max(Jc_s, R_s) if n <= 12 else (np.nan, np.nan)

    return dict(
        n_ok=len(Js),
        true_absc=np.max(true_absc),
        true_ratio_med=np.median([r for r in true_ratio if np.isfinite(r)]),
        bend=max(np.max(bend_hi), abs(np.min(bend_lo))),
        gersh=np.max(gersh),
        rohn=max(abs(rohn_hi), abs(rohn_lo)),
        hertz=max(abs(hz[0]), abs(hz[1])),
    )


print(f"{'system':<14} {'believed':<22} {'N_expl(true)':>13} {'bendixson':>11} "
      f"{'gershgorin':>11} {'rohn':>10} {'hertz':>10} {'rate-ratio':>11}")
print("-" * 108)
for s in SYSTEMS:
    r = run(s)
    if r is None:
        print(f"{s['name']:<14} all samples non-finite"); continue
    T = s["T"]
    rr = (s["rates"][1] / s["rates"][0]) if s["rates"] else float("nan")
    print(f"{s['name']:<14} {s['believed']:<22} {T*r['true_absc']:13.2e} "
          f"{T*r['bend']:11.2e} {T*r['gersh']:11.2e} {T*r['rohn']:10.2e} "
          f"{T*r['hertz']:10.2e} {rr:11.2e}")
print()
print("All columns are N_explicit = T_span * (bound on |Re lambda|):")
print("the number of steps an explicit method needs for STABILITY.")
