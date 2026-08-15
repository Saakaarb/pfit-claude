---
topic: The three user functions — their contract, conventions, and how to generate a skeleton
consumed_by: [pfit-skeleton, pfit-from-source, pfit-check, pfit-jax]
generated: false
owns: >
  Function names and signatures, the pseudocode conventions, what each function
  must return, the loss-normalization requirement, parameter ordering, and the
  rules for generating an unpopulated skeleton.
---

# The user_model.py contract

`user_model.py` holds three functions the user writes as **numpy-style
pseudocode**, not JAX. It is never executed as-is; `/pfit-jax` translates it
(see `jax_translation.md`).

Templates: `lib/utils/user_model_sample_unpopulated.py` (skeleton) and
`lib/utils/user_model_sample_populated.py` (the Robertson system, populated).

## Names and signatures

The three function names are fixed and must never be changed:

```python
def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval)
def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters)
def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters)
```

Key to the shapes: `Nts` = number of time steps, `Ny` = number of state
variables, `N_col` = number of columns in the dataset including time.

- `y` — state vector, shape `[Ny]`
- `dataset` — shape `[Nts, N_col-1]`; **time is excluded**, so column `k` of
  `dataset` is data column `k+1` of the CSV
- `t_eval` — evaluation times, shape `[Nts]`

## Conventions

- `trainable_parameters` and `fixed_parameters` are treated as **dicts** whose
  keys are the names in the config. This is a convenience for the user; the
  translation step converts the trainable dict to vector indexing.
- **Only `numpy` and `math` may be used.** No scipy, torch, or anything else.
- The parameter *order* in the config is load-bearing and is restated as a comment
  in the skeleton. Never reorder.
- `dataset`, `t_eval` and the initial conditions always describe **one
  experiment** — see the multi-experiment model in `project_context.md`.

## What each function must do

### `user_defined_system`

Returns a list/array of derivatives for **every** integrated variable, in the
config's variable order. Every integrated variable must have a derivative defined
and returned.

### `_compute_loss_problem`

Returns a **scalar** (or a 1-D array of length 1).

**The loss must be normalized so the value very likely falls between 0 and 1.**
The framework does not normalize it. An unscaled loss degrades gradient step
sizes and convergence. The standard pattern is a per-column-scaled RMSE:

```python
scale_factor = np.max(np.abs(dataset), axis=0)
scale_factor = np.where(scale_factor == 0, 1.0, scale_factor)
loss = np.sqrt(np.mean(np.square(
    (solution[:, obs_indices] - dataset[:, col_indices]) / scale_factor[col_indices])))
```

It must NOT loop over or aggregate multiple datasets.

For observables sampled at different times, see `staggered_data.md`.

### `writeout_description`

Returns an array to be written to `result_solution_expN.csv`. The useful shape
is `[time | data columns | solution columns]`, sized for plotting data against
fit. It must NOT loop over multiple datasets.

## Generating a skeleton (used by /pfit-skeleton)

Build the skeleton from `lib/utils/user_model_sample_unpopulated.py`, using the
config to populate:

- trainable parameter names from `model.trainable_parameters`, with a
  comment giving their vector order;
- fixed parameter names from `model.fixed_parameters`;
- integrated variable names and initial values from
  `model.integrated_variables`.

Rules:

- **Do not leave any function empty.** Provide stubs with comments showing what
  the user must fill in.
- **Preserve every comment** from the template.
- Add `import numpy as np` at the top.
- Output pure Python — no boilerplate prose, no markdown fences.
- When the config has multiple `experiments` blocks, add a comment below each
  function's argument list: `# dataset and t_eval represent ONE experiment's
  data; the framework calls this function once per experiment`.

A fully-populated model (used by `/pfit-from-source`) follows the same contract,
with the ODE right-hand side implemented exactly as the source specifies.
