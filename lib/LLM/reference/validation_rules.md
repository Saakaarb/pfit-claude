---
topic: What /pfit-check validates, the thresholds it uses, and the report format
consumed_by: [pfit-check]
generated: false
owns: >
  Every validation check and its severity, the optimizer-setting thresholds,
  and the structure and rules of the validation report.
---

# Validation rules

Applied by `/pfit-check` to `user_input.yaml` + `user_model.py`. The schema
is in `yaml_format.md`, the hard input constraints in `input_constraints.md`, and
the user-function contract in `user_model_contract.md` — this file says what to
*check* and how severe each failure is.

Severity:

- **Critical** — will CERTAINLY break JAX conversion, JIT compilation, or the
  run. The bar is high: only flag something critical if you are certain it fails
  on every run. Anything conditional on the data's values is a warning.
- **Warning** — MIGHT break one of those, or degrade convergence.

## Dataset checks

The framework imposes real structure on the CSV and enforces almost none of it.
Four of the failures below produce a **completed run with a plausible number**,
so they cannot be caught later — this section is the only place they are caught.

**Measure, never eyeball.** Run

```bash
./venv/bin/python3 tools/check_dataset.py <session>
```

which loads every CSV through `lib/utils/dataset_io.load_dataset` — the same
function the fit uses, so the two cannot disagree — and prints
one `PASS`/`FAIL`/`SKIP` line per id below, with the numbers behind it. Reading
the CSV yourself instead is not equivalent: a trailing delimiter and a duplicate
timestamp are both invisible to the eye. The tool reports facts and assigns no
severity; this table is the only authority on severity.

| id | Requirement | Severity | What happens if violated |
|---|---|---|---|
| D1 | >= 2 rows and >= 2 columns | critical | the array loads **1-D**, and `all_data[:, 0]` raises `IndexError` |
| D2 | no trailing delimiter on the rows | critical | appends a phantom **all-NaN column**, inflating the observable count and triggering D3x |
| D3 | NaN cells only when deliberate | informational | reported without a verdict; D3x is what decides it, and `staggered_data.md` supports NaN by design |
| D3x | if the data carries NaN, the loss uses a nan-safe reduction | critical | the loss is NaN for **every** candidate, sanitised to `error_loss`; the search flatlines with no error and a full log of constant cost |
| D4 | time column strictly increasing | critical | raises **inside JIT**; the traceback points at diffrax, not at the CSV, so it reads as a solver bug |
| D5 | no duplicate time values | warning | does **not** raise. The instant is saved twice and silently double-weighted in the loss |
| D6 | `t_eval[0] >= initial_time` when `initial_time` is set | critical | a save point before `t0` raises inside JIT |
| D7 | `t_eval[-1] > t0` | critical | does **not** raise. The solve returns `y0` as the entire trajectory, so the run completes having integrated nothing |
| D8 | every literal `dataset[:, k]` in the loss/writeout exists | critical | shape error at trace time, or a silently wrong observable when `k` happens to be in range |
| D8b | if the model differences `solution` against the whole `dataset`, the observable count equals the state dimension | critical | broadcast error, or silent broadcasting against every state when one side has width 1 |
| D9 | all experiments share column count **and** column order | critical | the mapping is positional and nothing in the config remaps it, so a mismatch fits a different observable per file |
| D10 | data row 0 agrees with the `init_val` of each **directly observed** state | warning | an irreducible loss floor the optimizer cannot remove — but only in regime 1; see below |
| D11 | which initial-condition regime the experiment is in | informational | never a failure; it decides whether D10 is a defect or expected |
| D12 | the `columns` block declares one entry per column of the file | critical | a column added to or removed from the CSV shifts every index the loss uses, silently |
| D13 | a header row, if present, agrees with `columns` | warning | two statements of the same column map, drifted apart |
| D14 | `model.observables` equals the keys `_observables` returns | critical | a column names a model quantity that the model does not compute, or computes under another name |
| D15 | every declared observable is referenced by some column | warning | usually a column whose `observes` was left off |

Notes on the two that need judgement:

- **D8/D8b are read off `user_model.py`**, so a `SKIP` means the indices are not
  literals, not that the check passed. Say so rather than reporting it clean.
- **D10 is computed only where `observes` says it can be.** The tool compares
  each directly-observed state's `init_val` against data row 0 and scores the gap
  as a fraction of that column's observed range — the range being what the loss
  normalises by, so the number is the share of the signal that is permanently
  mis-fit. A **derived** column has no state to compare against and is reported
  as unchecked, not as passing. Flag a disagreement; do not correct it.
  A deliberate offset (an equilibration period before `t_eval[0]`, which D11
  reports as regime 2) is valid — see `yaml_format.md`.

Report a `FAIL` using this file's severity and the tool's own numbers as the
evidence line. Do not restate a check that passed.

### Loss review (L1-L4)

The same command ends with a review of how the loss is **constructed**, measured
from the data alone — no solve, no fitted parameters, so it is available before
the first fit and stays within the cold-start invariant. These are **facts, not
findings**: report the ones that bear on this session and say what they imply.

| id | Reports |
|---|---|
| L1 | each measurement column's scale, the largest/smallest ratio, and the term weights that would follow **if** residuals were squared and left unnormalised |
| L2 | uncertainty columns that are declared, and whether the loss appears to use them |
| L3 | how the samples distribute across each observable's own range |
| L4 | the spread of data scales across experiments, which are averaged unweighted |

How to read them:

- **L1 is the relative-importance measure.** A large ratio means the loss is
  dominated by one channel unless it normalises per column. The share is
  conditional on an L2-norm loss — a loss built on absolute deviation or in log
  space weights differently, so check which the model uses before quoting it.
- **L2 is a hint, not proof**: the loss reads uncertainty columns positionally,
  so this cannot be decided by name alone. A declared uncertainty the loss
  ignores discards information the user took the trouble to supply.
- **L3 is about where the information is.** Samples piled into one decile of an
  observable's range mean the loss is mostly scoring that regime; the
  interesting part of the trajectory may be carrying almost no weight. This is
  a property of the data and the loss's space, not of any fit.
- **L4**: experiment losses are averaged unweighted, so a scale spread is a
  weighting nobody chose.

## Readiness checks

Applied by `/pfit-run` immediately before a fit, measured by

```bash
./venv/bin/python3 tools/check_ready.py <session> --mode full
./venv/bin/python3 tools/check_ready.py <session> --mode gradient-only
```

The mode must match the entry point about to be used, because R6's severity depends on it.

| id | Requirement | Severity | What happens if violated |
|---|---|---|---|
| R1 | `inputs/user_input.yaml` exists | critical | there is no session to run |
| R2 | `generated/user_model.py` exists | critical | nothing to translate or fit |
| R3 | `generated_script.py`'s source stamp matches the current `user_model.py` and `user_input.yaml` | critical | **the fit imports the script and never reads the model.** A stale script fits the previous version of the equations, completes normally, and reports parameters for a model you no longer have |
| R4 | the validation report is newer than the config and the model | warning | `/pfit-check` last ran against different inputs, so its verdict may not describe what is about to run |
| R5 | the last validation reported zero critical errors | critical | those errors were never resolved |
| R6 | a completed seed run's `final_design_point.csv` exists (legacy flat output supported) | **critical under `--mode gradient-only`**, informational under `--mode full` | it is the seed `fit_gradient_only.py` reads, and it raises `FileNotFoundError` without one. Logs alone are not enough: a run that died before writing a design point leaves logs but no seed. Both run modes preserve prior artifacts and create new run directories |

R3 compares **content**, not timestamps. `/pfit-jax` stamps the script it writes
with a hash of each source taken after stripping comments and blank lines, so the
check is immune to `touch`, to a clone or copy that flattens every modification
time, and to comment-only edits that cannot change the translation.

A script written before stamping existed carries no stamp. R3 then falls back to
modification times **and says that it did** — a fallback that can be defeated by
any of the above, so treat a passing mtime comparison as weaker evidence than a
passing stamp. Re-running `/pfit-jax` replaces it with a real one.

## Config checks

### API validity (check against the generated digests, never from memory)

- `integrator`, if present, must appear in the solver table of
  `lib/LLM/api/diffrax.md`. Absent from the table -> **critical** (the table
  already excludes solvers that exist but cannot be used here; anything not
  listed either does not exist or cannot work with the framework's stepsize
  controller). If `integrator` is absent from the config there is no error — it
  defaults to `Kvaerno5`.
- `gradient_optimizer`, if present, must be one the framework supports
  (`lib/algorithms/NODE/classes.py`) -> otherwise **critical**.
- `algorithm`, if present, must be `PSO` or `DE` -> otherwise **critical**;
  anything else silently falls through to PSO.
- If a digest's header versions disagree with the installed packages it is
  stale — regenerate before relying on it (see `project_context.md`).

### experiments

- At least one must exist -> critical if missing.
- Each must have `data_file` -> critical if missing. If one is missing, flag
  it; do NOT invent a filename.
- Every referenced CSV must exist in `sessions/<session>/inputs/`. Existence is
  all that is checked here — the CSV's *structure* is the Dataset checks above,
  and they are not optional.
- Each each name under `initial_conditions` must match a variable in
  `model.integrated_variables` -> critical on mismatch.
- `initial_conditions` is optional per experiment; its absence means that
  experiment uses the global `init_val`s. Do NOT flag its absence.
- Different experiments having different initial conditions is expected and
  valid. Do NOT flag it.

### model.trainable_parameters

- Are the search ranges sensible?
- Parameters whose range spans many orders of magnitude must use
  `logscale: true`. **It is your job to flag this** — never ask the user to check
  it themselves.
- All names must be valid Python identifiers.
- **Duplicate names across trainable, fixed or integrated variables are the
  top-priority critical error** — the run raises immediately.

### model.fixed_parameters / model.integrated_variables

- Are the fixed values reasonable? Are the initial values sensible?
- Are the names pythonic and valid identifiers?

### population_opt

Read the actual numbers from the config and evaluate each of these explicitly:

| Condition | Severity | Why |
|---|---|---|
| `population_size` < 20 | critical | too few to explore the space |
| `population_size` > 1000 with no `population_opt.stepsize_rtol` | warning | runs the global search at tight gradient tolerances; very slow |
| `processors` > available CPU cores | warning | oversubscribing cores will not speed the fit up |
| `num_iters` < 5 | warning | very few iterations |
| any `population_opt.stepsize_rtol` tighter than the matching `stepsize_rtol` | warning | zero-order tolerances should be looser, not tighter |
| `population_size` x `num_iters` < 20 x N² (N = number of trainable params) | warning | search budget likely insufficient to find a good basin |

Rule of thumb for the budget check: `population_size` >= 10 x N and
`num_iters` >= 20. State the actual values and the implied budget in the
warning so the user can decide.

### gradient_opt

| Condition | Severity | Why |
|---|---|---|
| `num_iters` < 3 | warning | very few gradient iterations |
| `max_steps` < 1000 | warning | many integrations may hit the step limit and score `error_loss` |
| `init_value_lr` < `end_value_lr` | critical | inverted LR schedule; loss diverges |
| any `stepsize_rtol`/`stepsize_atol` < 1e-12 | warning | near floating-point precision; may never converge |

**Optimizer choice governs iteration count and learning rate** — inspect
`gradient_optimizer`:

- `lbfgs` is quasi-Newton: it takes large curvature-informed steps, so
  a small `num_iters` (tens, even <10) is fine, and it performs its own line
  search, so the LR fields are irrelevant to it.
- `adam` (the default proposed for new sessions) is first-order and takes a
  **normalized** step: its update is
  `lr * m/(sqrt(v)+eps)`, and where the gradient sign is consistent that factor
  tends to ±1, so each step moves about `lr` in the scaled parameter space
  **regardless of the gradient's magnitude**. Since the framework scales every
  parameter to `[-1, 1]`, the distance adam can travel is about
  `num_iters * init_value_lr`.

  Warn if `num_iters < 1 / init_value_lr`, and state the implied travel
  distance. At the recommended `init_value_lr = 5e-3` that floor is **200**;
  `num_iters = 40` would move only 0.2 in a coordinate whose full range is 2,
  which cannot cross a basin. An annealing schedule lowers the real total below
  `num_iters * init_value_lr`, so treat the floor as optimistic.

  Also warn if `init_value_lr` > ~1e-2 (adam oscillates or diverges on the stiff
  ODE loss surface). The starting proposal is 1000 iterations, an initial rate
  of 5e-3, transition steps of 100, and annealing to ~1e-5.

  Clearing the floor is necessary, not sufficient: it bounds how far adam
  *could* move, not whether it converged. The exit-gradient ratio is the only
  evidence of that, and it is a post-fit check (S7 in `diagnosis_rules.md`).

## user_model.py checks

Treat the file as pseudocode throughout (see the "do not flag" list below).

### `user_defined_system`

- Every integrated variable has a derivative defined and returned -> critical.
- The code can be converted to JAX and JIT-compiled -> critical.
- Any parameter used but not defined?
- Any obvious logical errors? Are all defined parameters actually used?
- Only `numpy`/`math` used?

### `_compute_loss_problem`

- Returns a scalar, or a 1-D array of length 1 -> critical.
- The loss is normalized so it likely lies between 0 and 1 -> critical.
- Convertible to JAX and JIT-compilable -> critical.
- Must NOT loop over or aggregate multiple datasets -> critical if violated.
- Any undefined parameters? Obvious logical errors? Only numpy/math?

### `writeout_description`

- Returns an array.
- Must NOT loop over multiple datasets -> critical if violated.
- Any undefined parameters? Only numpy/math?

## Do NOT flag

1. That `trainable_parameters`/`fixed_parameters` are used as dicts — that is the
   intended pseudocode convention.
2. Import errors, or that the file would not run as-is. It is not meant to run.
3. Indentation or whitespace problems — those resolve in translation.
4. That the code "needs to be in JAX" or is JAX-incompatible/suboptimal. It is
   translated later. Never ask the user to convert it themselves.
5. Intentional blank/NaN cells, a `t=0` anchor row, or `np.isnan`/`np.nanmax` in
   the loss — see `staggered_data.md`.

## Report format

The report is the entire output. No boilerplate text around it.

Three sections, **Critical Errors**, then **Warnings**, then
**Recommendations**, each ordered most to least important. End with a line:

```
Number of critical errors: N
```

Rules:

- If a check passes, **omit it**. The report should be as small as possible and
  contain only what the user needs to or should change.
- If there are no critical errors, write `None detected` in that section.
- Where possible, tell the user how to remedy each point.
- When re-classifying an existing report, do not edit the wording of a point —
  only move it between sections or drop it. A point that is conditional on the
  data belongs in Warnings; a point that neither certainly nor possibly causes a
  failure should be removed entirely.

### The Recommendations section

Everything about this section — which recommendations exist, the evidence rule,
the entry format, the five-entry cap, and the fact that they are applied only on
the user's confirmation — is owned by `tuning_rules.md`. Do not restate its rules
here or work from memory.

The distinction this file owns: a **Warning** says an existing value may break or
degrade the run; a **Recommendation** proposes a better value based on evidence
from the model or the data. Never put the same point in both. Omit the section
entirely when `tuning_rules.md` yields nothing evidence-backed.
