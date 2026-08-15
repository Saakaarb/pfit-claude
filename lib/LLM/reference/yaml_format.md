---
topic: The user_input.yaml format — every section, field, default and valid value
consumed_by: [pfit-skeleton, pfit-new, pfit-check, pfit-jax]
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
  num_iters: 10
  stepsize_rtol: 1.0e-07
  stepsize_atol: 1.0e-09
  initial_timestep: 1.0e-06
  max_steps: 10000
  integrator: Kvaerno5
  gradient_optimizer: lbfgs
  init_value_lr: 1.0e-04
  end_value_lr: 1.0e-05
  transition_steps_lr: 2000
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
