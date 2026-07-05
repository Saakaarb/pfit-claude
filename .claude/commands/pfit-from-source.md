Generate both user_input.xml and a fully-populated user_model.py in a single step from an ODE system described in a paper, textbook, or any reference source.

## When to use

Use this instead of `/pfit-skeleton` when the user has a paper or reference that specifies the ODE system directly. Skips the blank-skeleton step and produces ready-to-check files.

## Steps

### Step 1 — Gather initial inputs

Ask the user for (if not provided as arguments):
- **Session name** — e.g. `glycolysis_session`
- **Reference** — a paper URL or local file path
- **Data file** — the CSV filename that already exists in `sessions/<session>/inputs/`

### Step 2 — Read the reference

- If the reference is a URL: use WebFetch to fetch it
- If it is a local path: use Read
- Extract all ODE systems, state variables, and parameters described in the paper

### Step 3 — Clarify the ODE system with the user (REQUIRED before proceeding)

Present your findings and ask the user to confirm before writing anything. Specifically:

1. **Which system to fit** — if the paper contains multiple ODE systems or model variants, list them and ask which one to use
2. **State variables** — list the variables you identified (names and physical meaning)
3. **Parameters** — list all parameters found in the equations, and your proposed split:
   - *Trainable* (to be fit): parameters the paper treats as unknown or estimated
   - *Fixed* (known constants): parameters the paper gives exact values for
   - Ask the user to confirm or adjust this split
4. **Initial conditions** — state the values you found in the paper; ask the user to confirm or correct
5. **Multi-experiment** — explicitly ask: "Do you have multiple datasets to fit simultaneously (e.g. the same system measured under different initial conditions or different experimental runs)?"
   - If yes: ask how many datasets, what CSV file each corresponds to, and whether the initial conditions differ between them (and if so, what the ICs are for each run)
   - If yes: the XML will need one `<EXPERIMENT>` block per dataset, with per-experiment `<INITIAL_CONDITIONS>` blocks where ICs differ
   - If no: a single `<EXPERIMENT>` block is sufficient

Wait for the user's response before proceeding.

### Step 4 — Read and clarify the data CSV(s)

For each CSV file the user identified (one for single-experiment, one per experiment for multi-experiment), read the first 20 rows.

Report for each file:
- Number of columns
- Column headers (if present) or index-based labels
- Sample values (min/max per column, or first few rows)
- Your best guess at what each column represents (time, which state variable, etc.)

Ask the user to confirm the column mapping (once — assume the same column layout applies to all files unless the user says otherwise):
- Which column index is time?
- Which column index corresponds to which state variable?
- Are any columns observables of the state (e.g. a derived quantity) rather than a direct state variable?

Wait for the user's response before proceeding.

### Step 5 — Ask for remaining settings

Present your best-guess defaults (derived below) and ask the user to confirm or override:

**Parameter search ranges:** For each trainable parameter:
- If the paper gives a value `v`, suggest `[v/100, v*100]` (log-scale if range > 2 orders of magnitude)
- If no value is given, suggest a physically reasonable range and flag it explicitly for the user to review

**Optimizer settings (suggest these defaults, adjustable by user):**
- Population-based: `POPULATION_SIZE = max(50, 10 × N_TRAINABLE)`, `NUM_ITERS = 20`, `PROCESSORS = 4`
- Gradient: `NUM_ITERS = 10`, `MAX_STEPS = 10000`, `INITIAL_TIMESTEP = 1e-6` (adjust if the problem timescale is very different from 1), `INIT_VALUE_LR = 1e-4`, `END_VALUE_LR = 1e-5`, `TRANSITION_STEPS_LR = 2000`, `DECAY_RATE_LR = 0.9`
- `STEPSIZE_RTOL` and `STEPSIZE_ATOL`: one value per integrated variable (e.g. `1e-7,1e-7` for 2 variables)

Wait for the user's response before proceeding.

### Step 6 — Generate user_input.xml

Using all confirmed information, generate a complete XML matching the format in `lib/utils/user_input_sample.xml`.

Rules:
- `N_TRAINABLE_PARAMETERS` must equal the count of `<PARAM>` blocks in `TRAINABLE_PARAMETER_DESCRIPTION`
- Set `LOGSCALE = Y` for any parameter whose MIN_VAL and MAX_VAL span more than 2 orders of magnitude
- `STEPSIZE_RTOL` and `STEPSIZE_ATOL` must each have exactly one value per integrated variable, comma-separated
- Use Python-compatible identifiers for all names (no spaces, no hyphens)
- `COLUMN_INFO` is optional; include it if the user's column mapping is useful context

**Multi-experiment:** generate one `<EXPERIMENT>` block per dataset. Include an `<INITIAL_CONDITIONS>` sub-block inside each `<EXPERIMENT>` only for variables whose IC differs from the global `INIT_VAL`. Omit the sub-block entirely for experiments that use the global ICs — do not add an empty `<INITIAL_CONDITIONS>` block. `VAR/NAME` values inside `<INITIAL_CONDITIONS>` must exactly match names in `INTEGRATED_SYSTEM_DESCRIPTION`.

Write to: `sessions/<session>/inputs/user_input.xml`

### Step 7 — Generate user_model.py

Generate a **fully-populated** (not skeleton) `user_model.py` using the structure from `lib/utils/user_model_sample_populated.py` as a template.

> **Staggered/ragged data:** if the source reports different observables at different (non-shared) time points, read `lib/LLM/staggered_data_instructions.txt` BEFORE building the data CSV and the loss. In brief: build one CSV on the union of all times with blank cells for unmeasured (observable, time) pairs (→ NaN), add a t=0 anchor row if the ICs precede the first sample, and mask NaNs in `_compute_loss_problem` with the sanitise-before-divide pattern. Do NOT split each observable into its own file.

#### `user_defined_system`
- Unpack every trainable parameter from the `trainable_parameters` dict (keys match XML names)
- Unpack every fixed parameter from the `fixed_parameters` dict
- Unpack state variables from `y` in XML order with descriptive names
- Implement the ODE RHS exactly as written in the paper
- Return `np.array([dy1dt, dy2dt, ...])` with a derivative for EVERY integrated variable

#### `_compute_loss_problem`
- Compute a normalized RMSE loss against the relevant data columns (using the confirmed column mapping)
- The `dataset` argument has shape `[Nts, N_col-1]` (time is excluded; columns are 0-indexed from after the time column)
- Scale by the max value per column so the loss is likely between 0 and 1:
  ```python
  scale_factor = np.max(np.abs(dataset), axis=0)
  scale_factor = np.where(scale_factor == 0, 1.0, scale_factor)
  loss = np.sqrt(np.mean(np.square((solution[:, obs_indices] - dataset[:, col_indices]) / scale_factor[col_indices])))
  ```
- Return a scalar for **one experiment** — do NOT loop over or aggregate multiple datasets; the framework handles that automatically

#### `writeout_description`
- Return a combined array: `[time | data columns | solution columns]`
- Size it to match what is useful for plotting: time + all observed columns from data + corresponding solution columns
- Handles one experiment at a time — do NOT loop over multiple datasets

Write to: `sessions/<session>/generated/user_model.py`
Create the `generated/` directory if it does not exist.

### Step 8 — Report to user

Tell the user:
- What was generated (parameter names, variable names, equation summary)
- Any assumptions made (ranges you guessed, initial conditions inferred, etc.)
- "Review both files, then run `/pfit-check <session>` to validate and auto-correct before JAX generation."

## Important rules

- Do NOT write any files until after the clarification rounds (Steps 3–5) are complete
- The data CSV is always user-provided — never generate or modify it
- Treat `trainable_parameters` and `fixed_parameters` as dicts in the generated pseudocode (same convention as the rest of the pipeline)
- Only numpy and math imports are allowed in user_model.py
- No boilerplate text in user_model.py — pure Python only
- All variable and parameter names must be valid Python identifiers
- If the paper is inaccessible (fetch fails, PDF unreadable), tell the user immediately and ask them to paste the relevant equations directly
