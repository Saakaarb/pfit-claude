Validate and auto-correct the user_model.py and user_input.xml for the current session.

## Steps

1. Ask the user for the session name if not provided as an argument. The session directory is `sessions/<session_name>/`.

2. Read the following files:
   - `sessions/<session_name>/inputs/user_input.xml`
   - `sessions/<session_name>/generated/user_model.py`
   - `lib/LLM/user_file_check_instructions.txt` — validation rules
   - `lib/LLM/inputs_fix_instructions.txt` — correction rules
   - `lib/LLM/model_output_check_inputcheck_instructions.txt` — report formatting rules
   - `lib/LLM/staggered_data_instructions.txt` — read if the data is staggered/ragged. Do NOT flag intentional blank/NaN cells, a t=0 (IC-time) anchor row, or `np.isnan`/`np.nanmax` in the loss as errors.

3. **Validate** both files by carefully applying every check in `user_file_check_instructions.txt`, plus the optimizer settings checks listed at the bottom of this file. For optimizer settings, read the actual numerical values from the XML and evaluate each check against them explicitly. Produce a report with two sections:
   - **Critical Errors** — will certainly cause downstream failure (JAX compilation, runtime, integration)
   - **Warnings** — may cause issues with convergence or correctness
   - End with: `Number of critical errors: N`

4. **If there are critical errors**, apply the correction rules from `inputs_fix_instructions.txt`:
   - Make MINIMAL changes to fix as many critical errors as possible
   - Do NOT change the logic of the code unless it is fundamentally broken
   - Do NOT change column index mappings
   - Edit `sessions/<session_name>/inputs/user_input.xml` with any XML fixes
   - Edit `sessions/<session_name>/generated/user_model.py` with any Python fixes

5. **Re-validate** after corrections. Repeat the validation → correction loop until either:
   - There are no critical errors, OR
   - You have iterated 3 times and cannot resolve remaining critical errors

6. Write the final validation report to `sessions/<session_name>/generated/user_input_check.txt`. The report should be concise — only include items the user needs to act on. If a check passes, do not mention it.

7. Tell the user the outcome:
   - If no critical errors: "Validation passed. Run `/pfit-jax` to generate the JAX optimization code."
   - If unresolvable critical errors remain: list them clearly and ask the user to fix them manually before re-running `/pfit-check`.

## Validation Checklist Summary (from user_file_check_instructions.txt)

**XML checks:**
- Parameter search ranges are sensible
- Parameters spanning many orders of magnitude use LOGSCALE = Y
- All names are valid Python identifiers
- No duplicate names across trainable, fixed, or integrated variables

**Optimizer settings checks (inspect actual values):**

Population-based (zero-order) optimizer:
- `POPULATION_SIZE` < 20 → critical (too few to explore the space)
- `POPULATION_SIZE` > 1000 and no `POP_STEPSIZE_RTOL` set → warning (will use tight gradient tolerances; likely very slow)
- `PROCESSORS` > number of available CPU cores → warning (no hard cap, but oversubscribing cores beyond `os.cpu_count()` will not speed up and may slow down the fit)
- `NUM_ITERS` < 5 → warning (very few iterations)
- If `POP_STEPSIZE_RTOL` is set and any value is tighter (smaller) than the corresponding `STEPSIZE_RTOL` value → warning (zero-order tolerances should be looser than gradient tolerances, not tighter)
- Assess whether `POPULATION_SIZE` and `NUM_ITERS` are adequate for the search space dimension (`N_TRAINABLE_PARAMETERS`). A reasonable rule of thumb: `POPULATION_SIZE` ≥ 10 × N and `NUM_ITERS` ≥ 20. If the product `POPULATION_SIZE × NUM_ITERS` is below 20 × N², flag a warning that the search budget may be insufficient to reliably find a good basin. State the actual values and the implied budget in the warning so the user can make an informed decision.

Gradient:
- `NUM_ITERS` < 3 → warning (very few gradient iterations)
- `MAX_STEPS` < 1000 → warning (low; many integrations may hit the step limit and return error_loss)
- `INIT_VALUE_LR` < `END_VALUE_LR` → critical (learning rate schedule is inverted; loss will diverge)
- Any `STEPSIZE_RTOL` or `STEPSIZE_ATOL` value < 1e-12 → warning (extremely tight; near floating-point precision, may never converge)
- **Optimizer-specific iteration/LR guidance (inspect `GRADIENT_OPTIMIZER`):**
  - `lbfgs` (default) is quasi-Newton — it takes large, curvature-informed steps, so a SMALL number of iterations is fine (tens, even <10). The learning-rate settings are largely irrelevant for L-BFGS.
  - `adam` is a first-order method — it takes many small steps, so it needs a LARGE `NUM_ITERS` (hundreds to ~1000) AND a modest, annealing learning rate. Flag a warning if `GRADIENT_OPTIMIZER = adam` and either `NUM_ITERS` < ~200 (too few Adam steps to converge) or `INIT_VALUE_LR` > ~1e-2 (too high — Adam will oscillate/diverge on the stiff ODE loss surface; start around 1e-3 and anneal down, e.g. to 1e-5).

**user_defined_system checks:**
- Every integrated variable has a derivative defined and returned
- No undefined parameters used
- Only numpy/math libraries used (no scipy, torch, etc.)

**_compute_loss_problem checks:**
- Returns a scalar (or 1D array of length 1)
- Loss is normalized to likely fall between 0 and 1
- Only numpy/math libraries used
- Must NOT loop over or aggregate multiple datasets (the framework handles per-experiment dispatch) — critical if violated

**writeout_description checks:**
- Returns an array
- No undefined parameters used
- Must NOT loop over or aggregate multiple datasets — critical if violated

**Multi-experiment checks (when XML has more than one EXPERIMENT block):**
- Every CSV filename referenced in `<FILENAME_DATA>` must exist in `sessions/<session>/inputs/`
- Every `VAR/NAME` inside each `<INITIAL_CONDITIONS>` block must exactly match a variable name in `INTEGRATED_SYSTEM_DESCRIPTION` — mismatch is a critical error
- The user functions (`_compute_loss_problem`, `writeout_description`) must not contain dataset loops or cross-experiment aggregation

## Important Rules
- Treat user code as pseudocode — do NOT flag missing JAX imports or non-JAX syntax
- Do NOT flag indentation errors
- Do NOT suggest the user convert code to JAX format themselves
- If no issues exist in a section, omit that section from the report
