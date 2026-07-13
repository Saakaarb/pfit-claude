# Sneyd IP3-Receptor Gating — pfit-claude session

## Reference
Sneyd & Dufour (2002), *PNAS* 99(4):2398–2403, DOI
[10.1073/pnas.032281999](https://doi.org/10.1073/pnas.032281999).
Ported from the **PEtab Benchmark-Models** collection (`Sneyd_PNAS2002`).

## Problem
A 6-state Markov model of the IP3-receptor Ca²⁺ channel. The observable is the
channel **open probability**, measured as a ~1 s transient at **9 clamp
conditions** — a dose response over fixed IP₃ ∈ {3,5,10} µM and Ca²⁺ ∈
{0.1…10} µM (15 points each). This is the **multi-experiment** example: one
shared **14-parameter** set fitted simultaneously across all 9 conditions.

## Equation system
6 channel states (R rest, O open, A activated, I₁/I₂ inactivated, S shut),
conserved (Σ = 1). Transitions follow the Sneyd kinetic laws driven by the
clamped Ca²⁺ and IP₃, e.g.

```
R→O : [(k2·L3 + l4·Ca)/(L3 + Ca(1+L3/L1))]·IP3 ;   O→A : [((k4·L5+l6)·Ca)/(L5+Ca)]
R→I1: [((k1·L1+l2)·Ca)/(L1+Ca(1+L1/L3))]      ;   O→S : [k3·L5/(L5+Ca)]
I1→R: (k_1 + l_2)                              ;   S→O : k_3   (reverse rates)
```
plus the analogous activated-state pair (A→I2 fwd, I2→A at `k_1+l_2`; A→O
reverse). With derived constants `L1 = k_1·l2/(k1·l_2)`, `L3 = k_2·l4/(k2·l_4)`,
`L5 = k_4·l6/(k4·l_6)`, the 14 fitted rate constants (k1–k4, k_1–k_4, l2/l4/l6,
l_2/l_4/l_6) all enter either directly above or through L1/L3/L5. Per-condition **IP₃ and Ca are injected as two constant
"input" states** (dy/dt = 0) set through each experiment's `INITIAL_CONDITIONS`
— the framework's clean way to give each experiment a different fixed input.

## Observable & loss
`open_probability = (0.9·A + 0.1·O)^4`. Loss = **RMSE** (a probability is
already in [0,1]); the framework averages the per-condition losses.

## Parameters (published vs pfit-obtained)
| Param | Published | pfit | Param | Published | pfit |
|---|---|---|---|---|---|
| k1 | 3.73 | 6.17 | l2 | 0.940 | 1.43e-2 |
| k2 | 1.0e5 | 901 | l4 | 2.86 | 2.75 |
| k3 | 15.7 | 10.3 | l6 | 1.0e5 | 39.5 |
| k4 | 9.99e4 | 9.05e3 | l_2 | 0.348 | 5.95e-3 |
| k_1 | 0.924 | 1.37 | l_4 | 1.39e-2 | 0.419 |
| k_2 | 1.00e-3 | 5.39e-2 | l_6 | 1.0e-3 | 0.158 |
| k_3 | 1.91 | 1.92 | k_4 | 3.08e3 | 372 |

All 14 searched on a log scale over the PEtab bounds [1e-3, 1e5].

## Fit quality
| | pooled RMSE (open prob.) |
|---|---|
| published parameters | 0.023 |
| pfit-obtained | **0.023** |

Our fit **matches the published quality** across all 9 conditions
(`outputs/fit_vs_published.png` — curves overlap). The raw parameters differ
substantially: the published optimum pins k2/k4/l6 at the upper search bound
and k_2/l_6 at the lower one, because only the combinations L3 (≈2e-6) and L5
(≈3e6) are identifiable — classic **sloppiness** in an 8-decade × 14-D box.

## Solution approach (pfit-claude)
1. SBML → `user_model.py` (6-state RHS + 2 input-carrier states + open-prob
   observable) and a 9-`EXPERIMENT` `user_input.xml`, each experiment setting
   its IP₃/Ca clamp via `INITIAL_CONDITIONS`.
2. `/pfit-check` (0 critical) → `/pfit-jax` → `fit_parameters.py sneyd_ipr`.
3. **Differential Evolution** (pop **1000**) → **Adam** refinement (**1000
   steps**, LR 1e-3 annealing to 1e-5), `Kvaerno5`.

**Key lesson (now in the pfit-check guidance):** a first attempt with Adam at
30 steps / LR 1e-2 stalled at RMSE 0.037; increasing to 1000 Adam steps at a
lower annealing LR (plus the larger DE) reached 0.023. **L-BFGS needs few
iterations (quasi-Newton); Adam needs many steps at a modest learning rate.**
