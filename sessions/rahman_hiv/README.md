# Rahman HIV Transmission — pfit-claude session

## Reference
Rahman et al. (2016), *Mathematical Biosciences*, DOI
[10.1016/j.mbs.2016.07.009](https://doi.org/10.1016/j.mbs.2016.07.009).
Ported from the **PEtab Benchmark-Models** collection (`Rahman_MBS2016`).

## Problem
A population-level HIV transmission model with disease progression stages and
treatment. The single observable is overall **prevalence (%)**, measured yearly
for 23 years as the epidemic rises from 0.3 % to a ~18.9 % plateau. **9 rate
constants** are fitted; deaths, recruitment and treatment-coverage constants are
fixed. Single experiment, fixed initial populations.

## Equation system
7 compartments: `susceptible`, `infected_{normal,moderate,weak}`,
`treated_{normal,moderate,weak}` (time in years).

```
Ṡ   = recruitment − λ·S − μ_S·S
İ_n = λ·S − worsen_n·I_n − μ_{I_n}·I_n            (worsen: n→m→w; treat: I→T)
İ_m = worsen_n·I_n − worsen_m·I_m − μ_{I_m}·I_m
İ_w = worsen_m·I_m − treat_w·I_w − μ_{I_w}·I_w
Ṫ_n = improve_m·T_m − μ_{T_n}·T_n   ;  Ṫ_m = improve_w·T_w − improve_m·T_m − μ_{T_m}·T_m
Ṫ_w = treat_w·I_w − improve_w·T_w − μ_{T_w}·T_w
```

Force of infection with behavioural-change damping:
`λ = [(β_n·I_n + β_m·I_m + β_w·I_w + β_t·ΣT)/N] · exp(−bcr·Σinfected)`,
where `β_n = rel_n·β_m`, `β_w = rel_w·β_m`, `β_t = 0.04·β_m`, `N = total pop`.

## Observable & loss
`prevalence = (1 − susceptible/N)·100`. Loss = **peak-normalised RMSE**
(equivalent to the PEtab Gaussian objective, since the noise σ is constant).

## Parameters (published vs pfit-obtained)
| Parameter | Published | pfit-obtained |
|---|---|---|
| infected_normal_transmission_rate_relative | 427 | 7.51 |
| infected_moderate_transmission_rate | 2.93e-3 | 9.05e-2 |
| infected_weak_transmission_rate_relative | 993 | 56.1 |
| infected_weak_treatment_rate | 1.0e-4 | 8.72e-1 |
| infected_normal_worsen_rate | 0.665 | 0.167 |
| infected_moderate_worsen_rate | 7.98e-3 | 0.927 |
| treated_moderate_improve_rate | 0.136 | 1.46e-3 |
| treated_weak_improve_rate | 0.776 | 1.0e-4 |
| behavioural_change_rate | 1.34e-7 | 3.67e-7 |

All searched on a log scale over the PEtab bounds.

## Fit quality
| | scale-normalised RMSE |
|---|---|
| published parameters | 0.002 |
| pfit-obtained | **0.004** |

Both reproduce the prevalence curve essentially perfectly (see
`outputs/fit_comparison.png` and `fit_vs_published.png`). The fitted rate
constants differ from published by orders of magnitude yet give the same curve
— a **single aggregate observable cannot identify 9 stage-specific rates**
(sloppiness). The automatic post-fit diagnostic (`outputs/sloppiness_report.txt`)
confirms this: it flags several non-identifiable (flat) directions, i.e. many
parameter combinations the prevalence data cannot constrain.

## Solution approach (pfit-claude)
1. SBML → `user_model.py` (7-ODE RHS, derived force-of-infection, prevalence
   observable) + `user_input.xml`.
2. `/pfit-check` (0 critical) → `/pfit-jax` → `fit_parameters.py rahman_hiv`.
3. **Differential Evolution** (pop 600, `RANDOM_SEED=42` for reproducibility) →
   **Adam** refinement (1000 epochs, LR 1e-3 annealing to 1e-5), `Kvaerno5`.
   Well-behaved and fast; no blow-up, no stiffness pathologies. The many Adam
   epochs at a modest LR drive the gradient norm to ~7e-3 (well converged),
   giving RMSE 0.004.
