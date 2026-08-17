---
topic: Hard constraints the framework imposes on the two user-provided inputs
consumed_by: [pfit-new, pfit-check]
generated: false
owns: >
  The dataset CSV and config constraints that cause a crash or silently wrong
  results, and the non-fatal input-quality issues.
---

# Input constraints (dataset + config)

The data loader is `np.genfromtxt(delimiter=',')` in
`lib/utils/helper_functions.py`; the config is parsed by `lib/utils/yamlread.py`.
The field-by-field schema lives in `yaml_format.md`; this file covers only what
must be *true* of the inputs.

## Critical — crashes, or silently wrong results

### Dataset (CSV)

- **Comma-delimited plain text.** Tab/space-delimited or Excel files fail.
- **A header row is optional.** It is detected (a first row that does not parse
  as numbers) and skipped on load by `lib/utils/dataset_io.py`. When present it
  must agree with the experiment's `columns` block. Without one, `columns` is the
  only record of what each column means.
- **Column 0 is time, monotonically increasing.** It is the integration/save grid.
- **Rectangular.** Every row must have the same number of columns.
- **No accidental blank cells.** Missing values become `NaN`, which makes the
  loss `NaN` (sanitised to `1e10`), so the fit cannot progress. Deliberate blanks
  under the staggered-data workflow are a separate, supported case — see
  `staggered_data.md`.
- **File present and named correctly** at
  `sessions/<session>/inputs/<data_file>`.
- **Column order is positional and must match `user_model.py`.** Columns 1..N
  (after time) are consumed in file order; the loss and writeout index
  `dataset[:, k]` by position. Nothing in the config remaps them. A wrong
  column order silently fits the wrong data.
- **Every experiment must declare a `columns` block**, one entry per column of
  the file, and its length must equal the file's column count. This is the only
  machine-readable record of what each column means — see `yaml_format.md`.

### Config (user_input.yaml)

- **Ordering is load-bearing.** Trainable-parameter order defines the
  optimization vector's index order; integrated-variable order defines `y[]`
  indexing. Both must match the unpacking order in `user_model.py`. A mismatch
  does not raise — it silently fits the wrong quantity.
- **`stepsize_rtol`/`stepsize_atol` (in either optimizer section) need exactly
  one value per integrated variable**, or a single value that applies to all of
  them. A list of the wrong length is rejected when the file is read.
- **Names must be globally unique** across trainable + fixed + integrated
  variables (`check_name_uniqueness` raises) **and be valid Python identifiers**
  — no spaces or hyphens; they become dict keys and identifiers in generated
  code.
- **No name may be a YAML boolean.** Unquoted `on`, `off`, `yes`, `no`, `true`
  and `false` are parsed as booleans, not strings, so a parameter or variable
  with one of those names never arrives as a name at all. Rename it. (`Y` and
  `N` are safe.)
- **Numeric fields must parse** as float (`min_val`, `max_val`, `value`,
  `init_val`, the tolerances, `initial_timestep`, `initial_time`) or as int
  (`population_size`, both `num_iters`, `processors`, `max_steps`,
  `random_seed`).

  Note that YAML 1.1 reads `1e-7` and `5E3` as *strings*, not floats — a float
  literal needs a dot in the mantissa and a signed exponent (`1.0e-07`). The
  reader coerces either spelling, so neither breaks; write the canonical form
  anyway so the file means the same to other YAML tools. See `yaml_format.md`.
- **No unknown keys**, in any section or at the top level. Every one is an error
  — there is no silently-ignored corner of the format.
- **Required settings** (no safe default — missing is an error):
  `stepsize_rtol`, `stepsize_atol`, `max_steps`, `initial_timestep`, both
  `num_iters`, `population_size`, `processors`, and each trainable parameter's
  `min_val`/`max_val`/`logscale`.
- **At least one entry under `experiments`**, each with a `data_file`.
- **`initial_conditions` names must match** a name in
  `model.integrated_variables` exactly, or the read fails.
- **Every integrated variable needs an `init_val`**, and the variable count must
  equal the state dimension returned by `user_defined_system`.

## Non-critical — degrades fit quality, no crash

- **Initial conditions come from the config, not the data.** Integration starts
  at the config's `init_val`s while the data's first row is compared against the
  simulated initial state. If row 0 disagrees with those ICs for an observed
  variable, that mismatch is a fixed penalty the optimizer cannot remove — make
  them consistent.
- **The framework does not normalize the loss.** See `user_model_contract.md`.
- **Search ranges and log-scaling.** Ranges far too wide or narrow, or missing
  `logscale: true` when they span many orders of magnitude, slow or stall the
  global search.
- **Mismatched scales across experiments.** Losses are averaged unweighted, so
  one experiment can dominate.
