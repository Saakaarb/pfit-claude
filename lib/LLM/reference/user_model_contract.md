---
topic: The three user functions — their contract, conventions, and how to generate a skeleton
consumed_by: [pfit-new, pfit-check, pfit-jax]
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

Template: `lib/utils/user_model_sample_populated.py` (the Robertson system).

## Names and signatures

The three function names are fixed and must never be changed:

```python
def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval)
def _observables(solution, trainable_parameters, fixed_parameters)
def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters)
def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters)
```

`_observables` is required **only when some dataset column measures a derived
quantity** rather than a state. A model whose every measured column is a state
(each column declaring `observes: <state>`) does not need it.

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

### `_observables`

Returns a **dict** mapping each name in `model.observables` to the trajectory of
that quantity, shape `[Nts]`.

```python
def _observables(solution, trainable_parameters, fixed_parameters):
        O = solution[:, 0]
        A = solution[:, 4]
        return {"Po": (0.9 * A + 0.1 * O) ** 4}
```

Why it exists: the dataset holds *observables*, which are usually a non-invertible
function of the state — an open probability, a relative percentage, a prevalence.
A column can then declare `observes: Po`, and the link from data to model is an
exact name match instead of something only recoverable by reading the loss.

Rules:

- the returned keys must equal `model.observables` exactly — no more, no fewer;
- a name must not collide with a state, parameter or fixed parameter;
- `_compute_loss_problem` and `writeout_description` both read the quantity from
  here rather than recomputing the algebra, so the two cannot drift;
- a quantity the RHS also needs (a reaction rate, say) stays in its own helper,
  which `_observables` calls. Do not duplicate it.

It is pseudocode like the rest of the file: `/pfit-jax` inlines it into the
generated loss, so no framework code calls it.

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

