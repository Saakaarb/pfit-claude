---
topic: Hard constraints the framework imposes on the two user-provided inputs
consumed_by: [pfit-skeleton, pfit-from-source, pfit-check]
generated: false
owns: >
  The dataset CSV and XML constraints that cause a crash or silently wrong
  results, and the non-fatal input-quality issues.
---

# Input constraints (dataset + XML)

The data loader is `np.genfromtxt(delimiter=',')` in
`lib/utils/helper_functions.py`; the XML is parsed by `lib/utils/xmlread.py`.
The field-by-field schema lives in `xml_format.md`; this file covers only what
must be *true* of the inputs.

## Critical — crashes, or silently wrong results

### Dataset (CSV)

- **Comma-delimited plain text.** Tab/space-delimited or Excel files fail.
- **No header row.** `dtype=float` turns a text header into `NaN` and corrupts
  the data; the first row must already be numbers.
- **Column 0 is time, monotonically increasing.** It is the integration/save grid.
- **Rectangular.** Every row must have the same number of columns.
- **No accidental blank cells.** Missing values become `NaN`, which makes the
  loss `NaN` (sanitised to `1e10`), so the fit cannot progress. Deliberate blanks
  under the staggered-data workflow are a separate, supported case — see
  `staggered_data.md`.
- **File present and named correctly** at
  `sessions/<session>/inputs/<FILENAME_DATA>`.
- **Column order is positional and must match `user_model.py`.** Columns 1..N
  (after time) are consumed in file order; the loss and writeout index
  `dataset[:, k]` by position. `COLUMN_INFO` is NOT used to remap. A wrong
  column order silently fits the wrong data.

### XML

- **`N_TRAINABLE_PARAMETERS` must equal the number of `<PARAM>` blocks.**
- **Ordering is load-bearing.** Trainable-parameter order defines the
  optimization vector's index order; integrated-variable order defines `y[]`
  indexing. Both must match the unpacking order in `user_model.py`. A mismatch
  does not raise — it silently fits the wrong quantity.
- **`STEPSIZE_RTOL`/`STEPSIZE_ATOL` (and any `POP_STEPSIZE_*`) need exactly one
  value per integrated variable.**
- **Names must be globally unique** across trainable + fixed + integrated
  variables (`check_name_uniqueness` raises) **and be valid Python identifiers**
  — no spaces or hyphens; they become dict keys and identifiers in generated
  code.
- **`LOGSCALE` must be exactly `Y` or `N`.**
- **Micro-format:** every leaf is `<P> KEY = VALUE </P>` with exactly one `=`. A
  value containing `=` breaks parsing.
- **Numeric fields must parse:** `MIN_VAL`/`MAX_VAL`/`VALUE`/`INIT_VAL` as
  float; `N_TRAINABLE_PARAMETERS`/`POPULATION_SIZE`/`NUM_ITERS`/`PROCESSORS`/
  `MAX_STEPS` as int.
- **Required settings** (no safe default — missing means a crash):
  `STEPSIZE_RTOL`, `STEPSIZE_ATOL`, `MAX_STEPS`, both `NUM_ITERS`,
  `POPULATION_SIZE`, `PROCESSORS`, and each trainable parameter's
  `MIN_VAL`/`MAX_VAL`/`LOGSCALE`.
- **At least one `<EXPERIMENT>`**, each with a `FILENAME_DATA`.
- **`INITIAL_CONDITIONS` names must match** a name in
  `INTEGRATED_SYSTEM_DESCRIPTION` exactly, or `get_y0` raises.
- **Every integrated variable needs an `INIT_VAL`**, and the variable count must
  equal the state dimension returned by `user_defined_system`.

## Non-critical — degrades fit quality, no crash

- **Initial conditions come from the XML, not the data.** Integration starts at
  the XML `INIT_VAL`s while the data's first row is compared against the
  simulated initial state. If row 0 disagrees with the XML ICs for an observed
  variable, that mismatch is a fixed penalty the optimizer cannot remove — make
  them consistent.
- **The framework does not normalize the loss.** See `user_model_contract.md`.
- **Search ranges and log-scaling.** Ranges far too wide or narrow, or missing
  `LOGSCALE = Y` when they span many orders of magnitude, slow or stall the
  global search.
- **`COLUMN_INFO` is parsed but unused.** Do not rely on it to remap columns.
- **Mismatched scales across experiments.** Losses are averaged unweighted, so
  one experiment can dominate.
