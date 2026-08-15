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

which loads every CSV exactly as `lib/utils/helper_functions.py` does and prints
one `PASS`/`FAIL`/`SKIP` line per id below, with the numbers behind it. Reading
the CSV yourself instead is not equivalent: a trailing delimiter and a duplicate
timestamp are both invisible to the eye. The tool reports facts and assigns no
severity; this table is the only authority on severity.

| id | Requirement | Severity | What happens if violated |
|---|---|---|---|
| D1 | >= 2 rows and >= 2 columns | critical | the array loads **1-D**, and `all_data[:, 0]` raises `IndexError` |
| D2 | no trailing delimiter on the rows | critical | appends a phantom **all-NaN column**, inflating the observable count and triggering D3x |
| D3 | NaN cells only when deliberate | see D3x | on its own, informational — `staggered_data.md` supports NaN by design |
| D3x | if the data carries NaN, the loss uses a nan-safe reduction | critical | the loss is NaN for **every** candidate, sanitised to `error_loss`; the search flatlines with no error and a full log of constant cost |
| D4 | time column strictly increasing | critical | raises **inside JIT**; the traceback points at diffrax, not at the CSV, so it reads as a solver bug |
| D5 | no duplicate time values | warning | does **not** raise. The instant is saved twice and silently double-weighted in the loss |
| D6 | `t_eval[0] >= initial_time` when `initial_time` is set | critical | a save point before `t0` raises inside JIT |
| D7 | `t_eval[-1] > t0` | critical | does **not** raise. The solve returns `y0` as the entire trajectory, so the run completes having integrated nothing |
| D8 | every literal `dataset[:, k]` in the loss/writeout exists | critical | shape error at trace time, or a silently wrong observable when `k` happens to be in range |
| D8b | if the model differences `solution` against the whole `dataset`, the observable count equals the state dimension | critical | broadcast error, or silent broadcasting against every state when one side has width 1 |
| D9 | all experiments share column count **and** column order | critical | the mapping is positional and nothing in the config remaps it, so a mismatch fits a different observable per file |
| D10 | data row 0 agrees with the `init_val`s for observed variables | warning | an irreducible loss floor the optimizer cannot remove. **Judgement, not arithmetic** — see below |
| D11 | which initial-condition regime the experiment is in | informational | never a failure; it decides whether D10 is a defect or expected |

Notes on the two that need judgement:

- **D8/D8b are read off `user_model.py`**, so a `SKIP` means the indices are not
  literals, not that the check passed. Say so rather than reporting it clean.
- **D10 cannot be automated**: which CSV column maps to which state is known only
  to the loss body. The tool prints each dataset's row 0 beside the `init_val`s;
  you decide whether they agree, and only for variables the loss actually
  compares. A deliberate offset (an equilibration period before `t_eval[0]`, which
  D11 reports as regime 2) is valid — see the initial-condition semantics in
  `yaml_format.md`. Flag a disagreement, do not correct it.

Report a `FAIL` using this file's severity and the tool's own numbers as the
evidence line. Do not restate a check that passed.

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

- `lbfgs` (default) is quasi-Newton: it takes large curvature-informed steps, so
  a small `num_iters` (tens, even <10) is fine, and it performs its own line
  search, so the LR fields are irrelevant to it.
- `adam` is first-order and takes a **normalized** step: its update is
  `lr * m/(sqrt(v)+eps)`, and where the gradient sign is consistent that factor
  tends to ±1, so each step moves about `lr` in the scaled parameter space
  **regardless of the gradient's magnitude**. Since the framework scales every
  parameter to `[-1, 1]`, the distance adam can travel is about
  `num_iters * init_value_lr`.

  Warn if `num_iters < 1 / init_value_lr`, and state the implied travel
  distance. At the recommended `init_value_lr = 1e-3` that floor is **1000**;
  `num_iters = 200` would move only 0.2 in a coordinate whose full range is 2,
  which cannot cross a basin. An annealing schedule lowers the real total below
  `num_iters * init_value_lr`, so treat the floor as optimistic.

  Also warn if `init_value_lr` > ~1e-2 (adam oscillates or diverges on the stiff
  ODE loss surface). Suggest starting near 1e-3 and annealing to ~1e-5.

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
