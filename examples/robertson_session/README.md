# Robertson Stiff Kinetics — worked example

A textbook **stiff** ODE system (Robertson, 1966): a three-species autocatalytic
reaction whose rate constants span nine orders of magnitude. It is the canonical
smoke-test for a stiff integrator and for parameter recovery, because the true
constants are known exactly.

## System

Three concentrations `y = (y1, y2, y3)` (A → B → C), integrated from
`y(0) = (1, 0, 0)`:

```
dy1/dt = -k1·y1 + k3·y2·y3
dy2/dt =  k1·y1 - k3·y2·y3 - k2·y2^2
dy3/dt =  k2·y2^2
```

Three trainable rate constants, all searched on a log scale:

| Param | Search range | True value |
|-------|--------------|------------|
| k1    | `[…]` (log)  | 0.04       |
| k2    | `[…]` (log)  | 3.0×10⁷    |
| k3    | `[…]` (log)  | 1.0×10⁴    |

## Data & loss

Data (`inputs/robertson_data.csv`) are the three species sampled on a
logarithmically spaced time grid from ~10⁻⁶ s to ~10² s. The loss is a
normalised RMSE across the three species (so the tiny `y2 ~ 10⁻⁵` transient is
weighted comparably to the O(1) `y1`, `y3`).

## Optimizer configuration

- Global: PSO, population 100, 20 iterations (default algorithm).
- Gradient: L-BFGS, 10 iterations, through the stiff `Kvaerno5` solve.
- No `RANDOM_SEED` set → the PSO stage is stochastic run-to-run.

## Result (this redo)

**Combined loss `L = 4.7×10⁻⁴`** — an essentially exact fit. The true Robertson
constants are recovered to three significant figures:

| Param | Fitted | True |
|-------|--------|------|
| k1 | 4.008×10⁻² | 0.04 |
| k2 | 3.006×10⁷  | 3×10⁷ |
| k3 | 1.003×10⁴  | 1×10⁴ |

Identifiability (`outputs/sloppiness_report.txt`): **well-determined** — Hessian
spectrum spans only 1.6 decades with **no** flat directions. The stiffest
(best-constrained) combination is dominated by `k1`; the least-constrained by
`k3`.

See `outputs/fit_result.png` (fit vs. data for all three species) and
`outputs/sloppiness_spectrum.png`.

## Files

```
inputs/user_input.xml          fit configuration
inputs/robertson_data.csv      time-series data
generated/user_model.py        ODE RHS + loss (numpy pseudocode)
generated/generated_script.py  JAX-jittable version actually run
outputs/                       results (cleared and rewritten on each run)
plot_fit.py, plot_fit_config.yaml   regenerate outputs/fit_result.png
```

## Re-run

```bash
venv/bin/python3 fit_parameters.py examples/robertson_session
venv/bin/python3 examples/robertson_session/plot_fit.py   # refresh fit_result.png
```
