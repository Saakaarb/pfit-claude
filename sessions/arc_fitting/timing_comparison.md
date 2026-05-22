# Timing Comparison: Standalone scipy vs pfit-claude (JAX)

## Problem
ARC (Accelerating Rate Calorimetry) battery thermal runaway model.
3 state variables, 8 trainable parameters. Stiff Arrhenius ODE — rate spans
6 orders of magnitude over the experiment (354 K → 665 K).
Equivalent budget: 350 particles × 50 iterations = 17,500 evaluations.

## Standalone (this session — scipy DE + Nelder-Mead)

| Item | Value |
|------|-------|
| DE wall time | **297 s (4.9 min)** |
| NM wall time | **3712 s (61.9 min)** |
| **Total wall time** | **4008 s (66.8 min)** |
| Optimizer | scipy differential_evolution + Nelder-Mead |
| Workers | 8 (multiprocessing) |
| Population (DE) | 352 (popsize=44 × 8 params) |
| DE iterations | 50 |
| ODE solver (DE) | scipy Radau, rtol=1e-3, atol=1e-3 |
| ODE solver (NM) | scipy Radau, rtol=1e-7, atol=1e-9 |
| Final loss | 0.657 |

Note: the NM phase dominated (62 min) because tighter tolerances make each
Radau solve much more expensive on this stiff Arrhenius system.

## pfit-claude (examples/ — JAX diffrax + PSO + Adam)

Timing from `examples/ARC_fitting/outputs/pso_fitting.log` and `NODE_fitting.log`:

| Item | Value |
|------|-------|
| PSO wall time | **669 s (11.1 min)** |
| Gradient wall time | **171 s (2.8 min)** |
| **Total wall time** | **839 s (14.0 min)** |
| Optimizer | PSO (custom JAX) + Adam gradient |
| Workers | 8 |
| Particles | 350 |
| PSO iterations | 50 |
| Gradient iterations | 20 |
| ODE solver (PSO) | diffrax Kvaerno5, rtol=1e-3, atol=1e-3 |
| ODE solver (grad) | diffrax Kvaerno5, rtol=1e-7, atol=1e-9 |
| Final loss | 0.719 (PSO) → improved by gradient phase |

Per-iteration PSO timing: first iter 36.9 s (incl. JIT compile),
steady state ~2–7 s/iter (all 350 particles vmapped simultaneously).

## Speedup

**4.8× faster** for the same evaluation budget.

## Why much smaller speedup vs sliding_basepoint (229×)?

| Factor | Sliding basepoint | ARC |
|--------|-------------------|-----|
| Speedup | 229× | 4.8× |
| ODE stiffness | Moderate (k~1e6 spring) | Very high (Arrhenius, 6 OOM rate change) |
| Internal solver steps | Hundreds | Tens of thousands |
| Fraction of time in Python callbacks | High | Low (Fortran solver dominates) |
| Benefit of JAX vmap | Large (100% parallel) | Moderate (solver bottleneck) |

For the sliding basepoint, the ODE is fast to solve and the Python callback
overhead dominated — JAX eliminated that overhead completely (229× gain).

For ARC, the ODE solver itself (Fortran Radau / diffrax Kvaerno5) takes thousands
of internal steps due to stiffness. The Fortran solver in scipy is already
highly optimized; JAX's main advantage (eliminating Python overhead) matters
less when the bottleneck is the implicit solver's internal linear algebra.
JAX still wins via vmap (parallel particle evaluation), but the gap narrows.

## Parameter quality

Both approaches found different local minima — this is a rough landscape with 8
parameters spanning many orders of magnitude (A1 range: 1e8–1e25).

| Param | Standalone | pfit reference | ratio |
|-------|-----------|----------------|-------|
| Ea1 | 2.50e-19 | 2.61e-19 | 0.957 |
| h1 | 1.56e+02 | 1.04e+02 | 1.507 |
| A1 | 1.33e+16 | 6.20e+13 | 214.6 |
| A2 | 1.11e+16 | 5.78e+17 | 0.019 |
| Ea2 | 2.17e-19 | 2.54e-19 | 0.854 |
| h2 | 1.13e+02 | 2.38e+02 | 0.476 |
| m2 | 4.42 | 4.28 | 1.033 |
| n2 | 4.34 | 1.12 | 3.889 |

Standalone loss (0.657) is slightly lower than pfit PSO loss (0.719 before
gradient refinement), but the parameters differ substantially — evidence of
multiple local minima in this high-dimensional Arrhenius landscape.
