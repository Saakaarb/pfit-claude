# Boehm STAT5 Dimerisation — pfit-claude session

## Reference
Boehm et al. (2014), *J. Proteome Res.* 13(12):5685–5694, DOI
[10.1021/pr5006923](https://doi.org/10.1021/pr5006923).
Ported from the **PEtab Benchmark-Models** collection
(`Boehm_JProteomeRes2014`).

## Problem
Phosphorylation and dimerisation of STAT5A / STAT5B in BaF3-EpoR cells after
erythropoietin (Epo) stimulation. Three relative-phosphorylation readouts are
measured at 16 time points over 240 min. **6 rate constants** are fitted; all
other constants and the cell geometry are fixed. A single experiment.

## Equation system
8 species — cytoplasmic A, B and the dimers ApB/ApA/BpB, plus nuclear
nApA/nApB/nBpB — in two compartments (cyt = 1.4, nuc = 0.45). Epo enters as a
decaying input `Epo(t) = 1.25e-7 · exp(-k_deg · t)` and drives mass-action
phosphorylation/dimerisation; dimers are imported to the nucleus and exported
back as monomers. Representative balances:

```
Ȧ    = -2·k_phos·Epo·A²  - k_phos·Epo·A·B  + (nuc/cyt)(2·k_exp_homo·nApA + k_exp_hetero·nApB)
ȦpA  =    k_phos·Epo·A²  - k_imp_homo·ApA
ṅApA = (cyt/nuc)·k_imp_homo·ApA - k_exp_homo·nApA        (and analogously for B / hetero)
```

## Observables & loss
Three nonlinear relative-phosphorylation percentages built from the cytoplasmic
species and a fixed specificity constant `specC17 = 0.107`, e.g.
`pSTAT5A_rel = (100·pApB + 200·s·pApA)/(pApB + s·A + 2s·pApA)`.
Loss = per-observable **peak-normalised RMSE** (three channels weighted
comparably).

## Parameters (published vs pfit-obtained)
| Parameter | Published | pfit-obtained | Search range (log) |
|---|---|---|---|
| Epo_degradation_BaF3 | 2.70e-2 | 2.66e-2 | [1e-5, 1e5] |
| k_exp_hetero | 1.00e-5 | 1.00e-7 | [1e-5, 1e5] |
| k_exp_homo | 6.17e-3 | 6.97e-3 | [1e-5, 1e5] |
| k_imp_hetero | 1.64e-2 | 1.59e-2 | [1e-5, 1e5] |
| k_imp_homo | 9.77e4 | 3.53e6 | [1e-5, 1e5] |
| k_phos | 1.58e4 | 1.66e4 | [1e-5, 1e5] |

## Fit quality
| | combined peak-normalised RMSE |
|---|---|
| published parameters | 0.064 |
| pfit-obtained | **0.062** |

The two parameter sets give **near-identical trajectories** despite differing by
orders of magnitude in `k_exp_hetero` and `k_imp_homo` — a textbook example of
**parameter sloppiness** (the 3 observables constrain only certain combinations).
See `outputs/fit_comparison.png`.

## Solution approach (pfit-claude)
1. Extracted the SBML reactions → `generated/user_model.py` (numpy ODE RHS +
   observables + loss); config in `inputs/user_input.xml`.
2. `/pfit-check` → validated (0 critical). `/pfit-jax` → JAX `generated_script.py`.
3. `python fit_parameters.py boehm_stat5`: **Differential Evolution** global
   search (pop 600, recentered ±2-order ranges) → **Adam** gradient refinement,
   `Kvaerno5` stiff integrator. Verified against the published parameters
   (RMSE 0.064) as a reachability check before fitting.
