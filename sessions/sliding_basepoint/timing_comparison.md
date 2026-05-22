# Timing Comparison: Standalone scipy vs pfit-claude (JAX)

## Problem
Sliding basepoint 2-mass friction model. 6 state variables, 5 trainable parameters.
Equivalent budget: 500 particles × 20 iterations = 10,000 function evaluations.

## Standalone (this session — scipy DE + Nelder-Mead)

| Item | Value |
|------|-------|
| Wall time | **1845 s (30.8 min)** |
| Optimizer | scipy differential_evolution |
| Workers | 8 (multiprocessing) |
| Population | 500 (popsize=100 × 5 params) |
| Iterations | 20 |
| ODE solver | scipy LSODA (Python callback) |
| Final loss | 0.2766 |

## pfit-claude (examples/ — JAX diffrax + PSO)

Timing from `examples/sliding_basepoint/outputs/pso_fitting.log`:

| Item | Value |
|------|-------|
| Wall time (PSO) | **8.07 s** |
| First iteration (incl. JIT compile) | 4.54 s |
| Steady-state per iteration | ~0.186 s |
| Optimizer | PSO (custom JAX) |
| Workers | 8 |
| Particles | 500 |
| Iterations | 20 |
| ODE solver | diffrax Kvaerno5 (JIT-compiled) |
| Final loss (PSO) | 0.3157 |

*Note: gradient-based refinement (NODE_fitting.log) shows 20 iterations but no times recorded — likely sub-second.*

## Speedup

**229× faster** for the same evaluation budget.

## Why

scipy LSODA calls the Python ODE function at every internal step — each of the 10,000
particle evaluations involves thousands of Fortran→Python boundary crossings
(~184 ms/eval on average).

JAX compiles the entire ODE solver + RHS into a single XLA program, then vmaps it
across all 500 particles simultaneously. No Python interpreter at runtime. Each
effective evaluation costs ~0.8 ms.

The first JAX iteration (4.54 s) includes trace + XLA compilation time. Subsequent
iterations drop to ~0.19 s for all 500 particles combined.
