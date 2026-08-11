---
topic: What /pfit-check validates, the thresholds it uses, and the report format
consumed_by: [pfit-check]
generated: false
owns: >
  Every validation check and its severity, the optimizer-setting thresholds,
  and the structure and rules of the validation report.
---

# Validation rules

Applied by `/pfit-check` to `user_input.xml` + `user_model.py`. The XML schema
is in `xml_format.md`, the hard input constraints in `input_constraints.md`, and
the user-function contract in `user_model_contract.md` — this file says what to
*check* and how severe each failure is.

Severity:

- **Critical** — will CERTAINLY break JAX conversion, JIT compilation, or the
  run. The bar is high: only flag something critical if you are certain it fails
  on every run. Anything conditional on the data's values is a warning.
- **Warning** — MIGHT break one of those, or degrade convergence.

## XML checks

### API validity (check against the generated digests, never from memory)

- `INTEGRATOR`, if present, must appear in the solver table of
  `lib/LLM/api/diffrax.md`. Absent from the table -> **critical** (the table
  already excludes solvers that exist but cannot be used here; anything not
  listed either does not exist or cannot work with the framework's stepsize
  controller). If `INTEGRATOR` is absent from the XML there is no error — it
  defaults to `Kvaerno5`.
- `GRADIENT_OPTIMIZER`, if present, must be one the framework supports
  (`lib/algorithms/NODE/classes.py`) -> otherwise **critical**.
- `ALGORITHM`, if present, must be `PSO` or `DE` -> otherwise **critical**;
  anything else silently falls through to PSO.
- If a digest's header versions disagree with the installed packages it is
  stale — regenerate before relying on it (see `project_context.md`).

### EXPERIMENT blocks

- At least one must exist -> critical if missing.
- Each must have `FILENAME_DATA` -> critical if missing. If one is missing, flag
  it; do NOT invent a filename.
- Every referenced CSV must exist in `sessions/<session>/inputs/`.
- Each `INITIAL_CONDITIONS/VAR/NAME` must match a variable in
  `INTEGRATED_SYSTEM_DESCRIPTION` -> critical on mismatch.
- `INITIAL_CONDITIONS` is optional per experiment; its absence means that
  experiment uses the global `INIT_VAL`s. Do NOT flag its absence.
- Different experiments having different initial conditions is expected and
  valid. Do NOT flag it.
- `COLUMN_INFO` is optional and informational. Do NOT flag its absence.

### TRAINABLE_PARAMETER_DESCRIPTION

- Are the search ranges sensible?
- Parameters whose range spans many orders of magnitude must use
  `LOGSCALE = Y`. **It is your job to flag this** — never ask the user to check
  it themselves.
- All names must be valid Python identifiers.
- **Duplicate names across trainable, fixed or integrated variables are the
  top-priority critical error** — the run raises immediately.

### FIXED_PARAM_DESCRIPTION / INTEGRATED_SYSTEM_DESCRIPTION

- Are the fixed values reasonable? Are the initial values sensible?
- Are the names pythonic and valid identifiers?

### POPULATION_OPT

Read the actual numbers from the XML and evaluate each of these explicitly:

| Condition | Severity | Why |
|---|---|---|
| `POPULATION_SIZE` < 20 | critical | too few to explore the space |
| `POPULATION_SIZE` > 1000 with no `POP_STEPSIZE_RTOL` | warning | runs the global search at tight gradient tolerances; very slow |
| `PROCESSORS` > available CPU cores | warning | oversubscribing cores will not speed the fit up |
| `NUM_ITERS` < 5 | warning | very few iterations |
| any `POP_STEPSIZE_RTOL` tighter than the matching `STEPSIZE_RTOL` | warning | zero-order tolerances should be looser, not tighter |
| `POPULATION_SIZE` x `NUM_ITERS` < 20 x N² (N = number of trainable params) | warning | search budget likely insufficient to find a good basin |

Rule of thumb for the budget check: `POPULATION_SIZE` >= 10 x N and
`NUM_ITERS` >= 20. State the actual values and the implied budget in the
warning so the user can decide.

### GRADIENT_OPT

| Condition | Severity | Why |
|---|---|---|
| `NUM_ITERS` < 3 | warning | very few gradient iterations |
| `MAX_STEPS` < 1000 | warning | many integrations may hit the step limit and score `error_loss` |
| `INIT_VALUE_LR` < `END_VALUE_LR` | critical | inverted LR schedule; loss diverges |
| any `STEPSIZE_RTOL`/`STEPSIZE_ATOL` < 1e-12 | warning | near floating-point precision; may never converge |

**Optimizer choice governs iteration count and learning rate** — inspect
`GRADIENT_OPTIMIZER`:

- `lbfgs` (default) is quasi-Newton: it takes large curvature-informed steps, so
  a small `NUM_ITERS` (tens, even <10) is fine, and it performs its own line
  search, so the LR fields are irrelevant to it.
- `adam` is first-order: it needs many small steps. Warn if `NUM_ITERS` < ~200
  (too few to converge) or `INIT_VALUE_LR` > ~1e-2 (Adam oscillates or diverges
  on the stiff ODE loss surface). Suggest starting near 1e-3 and annealing to
  ~1e-5.

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
