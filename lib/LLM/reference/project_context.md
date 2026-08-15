---
topic: What this project is, how a session is laid out, and how to run it
consumed_by: [pfit-skeleton, pfit-new, pfit-check, pfit-jax, ad-hoc work]
generated: false
owns: >
  Project purpose, the stage pipeline, session directory layout, the
  multi-experiment execution model, post-fit diagnostics, reproducibility
  semantics, the Python environment, and the API-digest regeneration rule.
---

# Project context

## Role

You are a coding assistant helping a user fit unknown parameters of a system of
ODEs (or DAEs) to user-provided time-series data. The user supplies: the system
of equations, its initial conditions, which parameters to fit, a search range
per parameter, the data, how to compute the loss, and what to write out.

The LLM layer generates and validates code. The optimization pipeline itself is
plain Python — it is not run by the model.

## The pipeline

Parameters are fitted in two stages:

1. **Population / zero-order search** (PSO or DE) — global exploration.
2. **Gradient-based refinement** (NODE: L-BFGS or Adam via optax) — local polish,
   seeded from the best point of stage 1.

Stage boundaries matter for tolerances: the global search may run at looser ODE
tolerances (`population_opt.stepsize_*`) than the gradient stage, and falls back to the
gradient tolerances when those are unset.

## Workflows

There is one workflow. `/pfit-new` is the entry point whether or not the user
has a source document: with one it extracts the equations, without one it asks
for them. Either way it writes `user_input.yaml` and `user_model.py` **together**,
which is what guarantees their orderings agree.

```
/pfit-new  ->  /pfit-check  ->  /pfit-jax
  ->  ./venv/bin/python3 fit_parameters.py <session_name>
  ->  /pfit-diagnose
```

The user supplies three things: the data CSV(s), a description of the system
(paper, pasted equations, or prose), and answers to the clarification rounds.
They are never asked to hand-author `user_input.yaml`.

`/pfit-skeleton` is a side path, not a stage: it rebuilds `user_model.py` from a
config that already exists.

## Session layout

```
sessions/<session_name>/
├── inputs/
│   ├── user_input.yaml       <- written by /pfit-new; user tunes it thereafter
│   └── <data>.csv            <- USER PROVIDES: time-series data
├── generated/
│   ├── user_model.py         <- created by /pfit-new (populated), or /pfit-skeleton (stub)
│   ├── user_input_check.txt  <- created by /pfit-check
│   └── generated_script.py   <- created by /pfit-jax
└── outputs/                  <- created by fit_parameters.py
    ├── final_design_point.csv
    ├── result_solution_exp1.csv   <- one file per experiment
    ├── pso_fitting.log  (or de_fitting.log)
    ├── NODE_fitting.log
    ├── run_stdout.log            <- only when the live view runs; see below
    ├── sloppiness_report.txt
    └── sloppiness_spectrum.png
```

The directory names are overridable via the `paths` section.

## The live view

Run on a terminal, `fit_parameters.py` and `fit_gradient_only.py` raise a live
plot of best-so-far loss against iteration for both stages, reading the
iteration logs as they are written. While it is up the pipeline's own console
output is captured to `outputs/run_stdout.log` — it would otherwise fight the
in-place redraw — and the last lines are echoed back when the view comes down,
so the fitted parameters still land in the terminal.

It is display only: it never writes into a session and cannot affect a fit.
Off a terminal (piped, `nohup`, cron, the pytest suite) it does not engage at
all and the output is exactly what it always was. `PFIT_LIVE=0` disables it for
one run; `auto_attach: false` in `tools/live_fit_monitor.yaml` disables it for
good. That same file configures it, and `tools/live_fit_monitor.py` shows the
same view for a fit already running in another terminal.

## Multi-experiment execution model

**Understand this before generating or reviewing any code.**

Multiple `experiments` blocks mean the same parameter set is fitted
simultaneously against multiple datasets — e.g. the same system measured under
different initial conditions or in different runs. The framework handles all
aggregation:

- `_compute_loss_problem(constants, trainable_variables)` is called **once per
  experiment**, with that experiment's own `constants` (its `dataset`,
  `t_eval`, `init_cond`).
- The framework averages the scalar losses across experiments. The user function
  must return a scalar for **one** experiment. It must NOT loop over or
  aggregate multiple datasets.
- `_write_problem_result` is likewise called once per experiment; outputs are
  written as `result_solution_exp1.csv`, `result_solution_exp2.csv`, ...
- `constants["init_cond"]` already reflects that experiment's initial
  conditions (global defaults merged with per-experiment overrides).

`dataset`, `t_eval` and `init_cond` always describe **one experiment at a
time**. Never write code that loops over or concatenates several experiments
inside the user functions.

The generated script is identical whether there is one experiment or many; the
per-experiment dispatch lives entirely in `lib/utils/helper_functions.py`.

Caveat worth knowing: the cross-experiment aggregation is an **unweighted mean**,
so experiments whose losses differ greatly in magnitude do not contribute
equally.

## Post-fit diagnostics (automatic)

Every gradient-based run writes `outputs/sloppiness_report.txt` and
`outputs/sloppiness_spectrum.png`. It eigendecomposes the Hessian of the loss at
the best fit in log-parameter space — the Fisher-information / "sloppiness"
spectrum (Gutenkunst et al. 2007; Hass et al. 2019) — and reports:

- the eigenvalue spectrum and its spread (**sloppy** if it spans more than ~6
  orders of magnitude);
- the number of practically **non-identifiable** (near-zero-eigenvalue)
  directions;
- the stiffest / sloppiest eigenvectors — which parameter *combinations* the
  data does and does not constrain — plus a per-parameter participation score.

Properties: it never breaks a fit (wrapped in try/except); it auto-skips models
with more than 60 parameters (finite-difference Hessian cost); it tries
second-order autodiff and falls back to finite-differencing the gradient (the
fallback is what actually runs, since 2nd-order AD through the stiff solver is
unavailable); and it uses the framework loss's implied noise model — so the
spread and eigenvectors match the Gauss-Newton FIM, but **absolute confidence
intervals are not calibrated**. It is a LOCAL measure, meaningful only at a
converged optimum. Implementation: `lib/utils/sloppiness.py`.

Re-run stand-alone on a completed session without re-fitting:

```bash
./venv/bin/python3 analyze_fit.py <session_name>
```

## Reproducibility

Setting `random_seed` in `population_opt` makes the fit deterministic
(same machine, same library versions). It is threaded to every stochastic
component:

- **LHS initial sampling** (`get_lhs_sampling`, used by both PSO and DE) — passed
  as `seed=`. `scipy.stats.qmc` uses its own Generator, NOT NumPy's global
  state, so it must be seeded explicitly; `np.random.seed()` does not reach it.
- **PSO** — `np.random.seed(seed)` in `initialize_swarm`, covering both initial
  velocities and the per-iteration cognitive/social draws, which pyswarms takes
  from NumPy's global RNG.
- **DE** — the same seed is passed to `scipy.optimize.differential_evolution`.
  When `random_seed` is unset, DE falls back to `42`, so **DE is reproducible by
  default; PSO is not**.
- **Gradient stage** — already deterministic (optax lbfgs/adam have no RNG, and
  the initial guess is the fixed best point from the global search).

The parallel loss evaluation shards independently of `processors`, so the device
count does not change results. Determinism is bit-for-bit only on identical
hardware and identical library versions.

## Python environment

**Always** use the project venv for any Python command:

```bash
./venv/bin/python3
```

Never use the system `python`/`python3` — jax, diffrax and optax are installed
only in this venv.

## API digests

`requirements.txt` pins an old, mutually compatible set (jax 0.6.2 /
diffrax 0.7.2 / optax 0.2.8 / scipy 1.15.3) whose APIs differ from current
upstream documentation. `tools/gen_api_context.py` introspects the **installed**
packages and writes version-pinned digests to `lib/LLM/api/`.

Rules:

- **Never write a diffrax / jax / optax call, or name a solver or optimizer,
  from memory.** The digests are authoritative; anything not in them does not
  exist in the pinned versions.
- Every digest header carries a version stamp. If it disagrees with the
  installed packages the digest is **stale** — regenerate before trusting it.
- Regenerate after any change to `requirements.txt`:

  ```bash
  ./venv/bin/python3 tools/gen_api_context.py
  ```

  Settings and the curated gotchas live in `tools/gen_api_context.yaml`. The
  digests are generated artifacts and must never be hand-edited. The generator
  rewrites only files whose content changed, so a re-run on an unchanged
  environment leaves the git tree clean.
