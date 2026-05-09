Validate and auto-correct the user_model.py and user_input.xml for the current session.

## Steps

1. Ask the user for the session name if not provided as an argument. The session directory is `sessions/<session_name>/`.

2. Read the following files:
   - `sessions/<session_name>/inputs/user_input.xml`
   - `sessions/<session_name>/generated/user_model.py`
   - `lib/LLM/user_file_check_instructions.txt` — validation rules
   - `lib/LLM/inputs_fix_instructions.txt` — correction rules
   - `lib/LLM/model_output_check_inputcheck_instructions.txt` — report formatting rules

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

PSO:
- `NUM_PARTICLES` < 20 → critical (too few to explore the space)
- `NUM_PARTICLES` > 1000 and no `PSO_STEPSIZE_RTOL` set → warning (PSO will use tight gradient tolerances; likely very slow)
- `PROCESSORS` > 8 → warning (hard limit in fit_parameters.py)
- `NUM_ITERS` < 5 → warning (very few PSO iterations)
- If `PSO_STEPSIZE_RTOL` is set and any value is tighter (smaller) than the corresponding `STEPSIZE_RTOL` value → warning (PSO tolerances should be looser than gradient tolerances, not tighter)
- Assess whether `NUM_PARTICLES` and `NUM_ITERS` are adequate for the search space dimension (`N_TRAINABLE_PARAMETERS`). A reasonable rule of thumb: `NUM_PARTICLES` ≥ 10 × N and `NUM_ITERS` ≥ 20. If the product `NUM_PARTICLES × NUM_ITERS` is below 20 × N², flag a warning that the search budget may be insufficient to reliably find a good basin. State the actual values and the implied budget in the warning so the user can make an informed decision.

Gradient:
- `NUM_ITERS` < 3 → warning (very few gradient iterations)
- `MAX_STEPS` < 1000 → warning (low; many integrations may hit the step limit and return error_loss)
- `INIT_VALUE_LR` < `END_VALUE_LR` → critical (learning rate schedule is inverted; loss will diverge)
- Any `STEPSIZE_RTOL` or `STEPSIZE_ATOL` value < 1e-12 → warning (extremely tight; near floating-point precision, may never converge)

**user_defined_system checks:**
- Every integrated variable has a derivative defined and returned
- No undefined parameters used
- Only numpy/math libraries used (no scipy, torch, etc.)

**_compute_loss_problem checks:**
- Returns a scalar (or 1D array of length 1)
- Loss is normalized to likely fall between 0 and 1
- Only numpy/math libraries used

**writeout_description checks:**
- Returns an array
- No undefined parameters used

## Important Rules
- Treat user code as pseudocode — do NOT flag missing JAX imports or non-JAX syntax
- Do NOT flag indentation errors
- Do NOT suggest the user convert code to JAX format themselves
- If no issues exist in a section, omit that section from the report
