---
topic: The user_input.yaml format — every section, field, default and valid value
consumed_by: [pfit-new, pfit-check, pfit-jax]
generated: false
owns: >
  The YAML schema, which fields are required vs optional, their defaults, where
  the valid values for each come from, and the two YAML-specific traps
  (exponent literals, bool-like names).
---

# user_input.yaml format

Parsed by `lib/utils/yamlread.py`. Standard YAML — no custom syntax.

## Single experiment (most common)

```yaml
experiments:
  - data_file: mydata.csv

model:
  trainable_parameters:
    - {name: mu, min_val: 0.001, max_val: 100.0, logscale: true}
    - {name: k,  min_val: 0.1,   max_val: 10.0,  logscale: false}
  fixed_parameters:
    - {name: some_const, value: 5.0}
  integrated_variables:
    - {name: x1, init_val: 1.0}
    - {name: x2, init_val: 0.0}

population_opt:
  population_size: 100
  num_iters: 20
  processors: 8
  algorithm: PSO
  random_seed: 0
  stepsize_rtol: 1.0e-05
  stepsize_atol: 1.0e-07

gradient_opt:
  num_iters: 1000
  stepsize_rtol: 1.0e-07
  stepsize_atol: 1.0e-09
  initial_timestep: 1.0e-06
  max_steps: 10000
  integrator: Kvaerno5
  gradient_optimizer: adam
  init_value_lr: 5.0e-03
  end_value_lr: 1.0e-05
  transition_steps_lr: 100
  decay_rate_lr: 0.9

output:
  write_results: true
```

## Two YAML traps the format cannot hide

**1. Exponent literals.** PyYAML implements YAML 1.1, in which a float literal
needs a dot in the mantissa **and** a signed exponent. `1e-7`, `5E3` and `1e+7`
are parsed as *strings*; only `1.0e-07` is a float.

The reader coerces every numeric field explicitly, so both spellings work and
nothing breaks. But **write the canonical form** — `1.0e-07`, not `1e-7` — so
the file means the same thing to every other YAML tool that reads it.

**2. Names that YAML turns into booleans.** Unquoted `on`, `off`, `yes`, `no`,
`true` and `false` are booleans, as keys *and* values. A parameter or variable
with one of those names never reaches the reader as a name. It is rejected with
an explanation rather than silently mangled, but the fix is to rename it.
(`Y` and `N` are safe — they are not YAML booleans.)

## Declaring the dataset columns (required)

Every experiment needs a `columns` block. The CSV is consumed positionally as a
bare numeric matrix, so without this the meaning of each column is recorded
nowhere machine-readable — the map from state to observable otherwise lives as
arbitrary Python inside `_compute_loss_problem`.

```yaml
experiments:
  - data_file: run_A.csv
    columns:
      - {name: time, units: s}
      - {name: X, units: mol/L, observes: X}
      - {name: Z, units: mol/L, observes: Z}
      - {name: X_sd, units: mol/L, uncertainty_of: X}
      - {name: Z_sd, units: mol/L, uncertainty_of: Z}
```

Entry `i` describes column `i`, so the first entry is always the time column.

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | a valid identifier, unique within the experiment |
| `units` | no | free text, for the reader |
| `observes` | no | names an integrated variable this column measures **directly**. Omit for a derived observable |
| `uncertainty_of` | no | names another measurement column in the same experiment; marks this column as its standard deviation |

Rejected at read time: an `observes` or `uncertainty_of` naming something that
does not exist, either on column 0, an `uncertainty_of` pointing at another
uncertainty column or at itself, duplicate names, and fewer than two entries.

The block is **descriptive** — nothing at runtime reads it. What it buys:

- `/pfit-new` can propose a loss, and knows to weight by sigma when uncertainty
  columns exist;
- `/pfit-check` can compare each directly-observed state's `init_val` against
  data row 0 (D10), which is otherwise not decidable by tooling;
- adding or removing a CSV column becomes an error (D12) rather than a silent
  re-indexing of every observable.

### Optional header row

The CSV may carry a header. It is detected (a first row that does not parse as
numbers), skipped on load, and **must agree with `columns`** (D13). A header is
convenient because it is usually already in the user's file, but `columns` is
the authority; the header is a redundant second statement of the same thing.

## Multi-experiment

Add one entry to `experiments` per dataset. `initial_conditions` overrides the
global `init_val` for that experiment only; variables not listed inherit the
global value. See `project_context.md` for what multiple experiments *mean*.

```yaml
experiments:
  - data_file: run_A.csv
    initial_conditions:
      x1: 1.0
  - data_file: run_B.csv
    initial_conditions:
      x1: 2.0
# model, population_opt, gradient_opt identical to single-experiment
```

Rules specific to multi-experiment configs:

- Each entry needs its own `data_file` with a distinct CSV.
- `initial_conditions` is optional per experiment; omit the key entirely when
  that experiment uses the global ICs.
- Every name under `initial_conditions` must match a name in
  `model.integrated_variables`; an unknown name is rejected when the file is
  read.
- Different experiments having different initial conditions is expected and
  valid.

## Initial-condition semantics: `init_val` vs the data's row 0

These are two different things and the framework never reconciles them.

- **The initial condition comes only from the config.** `get_y0` builds `y0` from
  `model.integrated_variables[].init_val`, overridden per experiment by
  `experiments[].initial_conditions`. It never reads the CSV. Both must be plain
  numbers — **an initial condition cannot be a trainable parameter**, so the fit
  can never absorb an error in one.
- **Integration starts at `initial_time`**, defaulting to `t_eval[0]`.
- **Every data time is a save point** (`SaveAt(ts=t_eval)`), so `solution` row `k`
  is the state at `t_eval[k]` and is differenced against `dataset` row `k`.

Whether `init_val` may differ from data row 0 depends entirely on `initial_time`:

**Regime 1 — `initial_time == t_eval[0]` (or unset).** The solve starts exactly
where the first save point is, so `solution[0]` **is** `y0`, identically, for
every candidate parameter set. Any disagreement with data row 0 is therefore a
constant penalty no parameter can remove; it sets a floor on the achievable loss
and biases the fit as it tries to compensate downstream. Confirm it by reading
row 0 of `result_solution_expN.csv`: the simulated column will reproduce the
`init_val` to full precision.

**Regime 2 — `initial_time < t_eval[0]`.** The solver integrates over
`[initial_time, t_eval[0]]` before the first comparison, so `solution[0]` is the
**evolved** state, not `y0`. Here the two legitimately differ, and the gap is
something the parameters can explain rather than a fixed penalty.

Consequences for setup:

- Set `initial_time` below `t_eval[0]` whenever the true state at the first
  measurement is unknown and the system needs to equilibrate into it. That
  converts an unremovable offset into a fitted transient.
- `initial_time > t_eval[0]` is always an error: it asks for a save point before
  `t0` and raises inside JIT (check D6 in `validation_rules.md`).
- Unobserved states have no row-0 counterpart at all, so their `init_val` is a
  pure assumption. Say so when reporting; it is a common source of a fit that
  cannot be improved by any parameter.

## Optional paths section

```yaml
paths:
  user_input_dir: inputs
  generated_dir: generated
  output_dir: outputs
```

## Field reference

**Unknown keys raise in every section, and so does an unknown top-level
section.** There is no silently-ignored corner of this format.

### model

| Field | Required | Notes |
|---|---|---|
| `trainable_parameters` | yes | A **list**, and its order is the optimization vector's index order |
| `name`, `min_val`, `max_val`, `logscale` | yes, per parameter | `logscale` is a real boolean (`true`/`false`); `Y`/`N` are also accepted |
| `fixed_parameters` | no | list of `{name, value}` |
| `integrated_variables` | yes | A **list**, and its order is the `y[]` index order |
| `name`, `init_val` | yes, per variable | |

There is no declared parameter count: the length of `trainable_parameters` is
the count.

### population_opt

| Field | Required | Default | Valid values |
|---|---|---|---|
| `population_size` | yes | — | int |
| `num_iters` | yes | — | int |
| `processors` | yes | — | int (XLA host device count) |
| `algorithm` | no | `PSO` | `PSO`, `DE` |
| `random_seed` | no | unset | int |
| `stepsize_rtol` / `stepsize_atol` | no | falls back to the gradient tolerances | one value per integrated variable, or a single value applied to all of them |

### gradient_opt

New sessions default to Adam in the assistant workflow: explicitly write
`gradient_optimizer: adam`. The table below records parser defaults when fields
are omitted; its legacy `lbfgs` fallback is not the new-session recommendation.

| Field | Required | Default | Valid values |
|---|---|---|---|
| `num_iters` | yes | — | int |
| `stepsize_rtol` / `stepsize_atol` | yes | — | one value per integrated variable, or a single value applied to all of them |
| `initial_timestep` | yes | — | float |
| `max_steps` | yes | — | int |
| `integrator` | no | `Kvaerno5` | **see `lib/LLM/api/diffrax.md`** — the generated solver table is the only authority |
| `gradient_optimizer` | no | `lbfgs` | `lbfgs`, `adam` (case-insensitive; see `lib/LLM/api/optax.md` for their real signatures) |
| `initial_time` | no | `t_eval[0]` | float |
| `init_value_lr`, `end_value_lr`, `transition_steps_lr`, `decay_rate_lr` | no | — | float; consumed by `adam` only |

### output

| Field | Default | Valid values |
|---|---|---|
| `write_results` | `false` | `true` writes `result_solution_expN.csv` |
