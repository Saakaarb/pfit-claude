Generate a user_model.py skeleton for the current session from its user_input.xml.

## Steps

1. Ask the user for the session name if not provided as an argument (e.g. `vanderpol_session`). The session directory is `sessions/<session_name>/`.

2. Read the session's XML: `sessions/<session_name>/inputs/user_input.xml`

3. **Validate the framework inputs** (the dataset CSV(s) and the XML) against the **Input Constraints** section below, BEFORE generating anything:
   - Read the first ~20 rows of each experiment's data CSV (`sessions/<session_name>/inputs/<FILENAME_DATA>`).
   - Check **every** item in the Input Constraints checklist — do NOT stop at the first failure. Work through the entire list and collect all violations.
   - Report **all** findings (not just the first), grouped into **Critical** (will cause the fit to fail or produce silently wrong results) and **Non-critical** (may degrade fit quality). List every individual issue with the specific file/field it concerns. End the report with `Number of critical input issues: N`.
   - **Hard gate:** if one or more critical issues are found, STOP — do NOT generate or write `user_model.py` (or create any output). Present the complete list of critical issues (and any non-critical ones) and tell the user to fix the input files (or run `/pfit-check`) and re-run `/pfit-skeleton`. Only continue to the steps below when the number of critical issues is 0.
   - Non-critical issues do not block generation: report them as warnings and proceed.

4. Read these reference files to understand what to generate:
   - `lib/LLM/api/jax.md` — the `jnp` functions the pseudocode will later be
     translated to, and the tracing rules it must respect. The skeleton is
     numpy pseudocode, but writing it in a shape that has no JAX equivalent
     (boolean-mask indexing, in-place assignment, Python branching on values)
     guarantees a failure at `/pfit-jax` time.
   - `lib/LLM/user_model_generation_instructions.txt` — generation rules
   - `lib/utils/user_model_sample_unpopulated.py` — skeleton template
   - `lib/utils/user_model_sample_populated.py` — populated example (Robertson)
   - `lib/utils/user_input_sample.xml` — example XML (Robertson)
   - `lib/LLM/staggered_data_instructions.txt` — read if observables are sampled at different (non-shared) time points; covers the union-grid + NaN-masking layout, the t=0 anchor row, and why per-observable file splitting does NOT work

5. **Only proceed past this point if step 3 found 0 critical input issues.** Following the rules in `user_model_generation_instructions.txt` and using the Robertson example as a reference, generate a `user_model.py` skeleton populated with:
   - The correct trainable parameter names (from `TRAINABLE_PARAMETER_DESCRIPTION`)
   - The correct fixed parameter names (from `FIXED_PARAM_DESCRIPTION`)
   - The correct integrated variable names and initial conditions (from `INTEGRATED_SYSTEM_DESCRIPTION`)
   - Comments indicating the ordering of trainable parameters in the vector
   - All three function stubs: `user_defined_system`, `_compute_loss_problem`, `writeout_description`

6. Create the `sessions/<session_name>/generated/` directory if it doesn't exist.

7. Write the skeleton to `sessions/<session_name>/generated/user_model.py`.

8. If the XML contains multiple `<EXPERIMENT>` blocks, add a comment near the top of each function (below the argument list) stating: `# dataset and t_eval represent ONE experiment's data; the framework calls this function once per experiment`.

9. Tell the user what was generated and remind them to fill in the ODE logic in `user_defined_system`, the loss computation in `_compute_loss_problem`, and the writeout in `writeout_description` before running `/pfit-check`.

10. At the end, VERIFY your implementation by cross checking against the input XML again

## Rules (from user_model_generation_instructions.txt)
- Do NOT leave any function empty — include stubs with comments showing what to fill in
- Add `import numpy as np` at the top
- Do NOT add any boilerplate text; the file should be pure Python
- Preserve all comments from the template
- For multi-experiment XMLs: `dataset` and `t_eval` always represent a single experiment; do NOT generate loops over multiple datasets inside any of the three functions

## Input Constraints (dataset + XML)

These are the assumptions the framework imposes on the two user-provided inputs. The data loader is `np.genfromtxt(delimiter=',')` in `lib/utils/helper_functions.py`; the XML is parsed by `lib/utils/xmlread.py`.

### Critical — will cause the fit to fail, or produce silently wrong results

**Dataset (CSV):**
- **Comma-delimited plain text.** Tab/space-delimited files or Excel files fail (`delimiter=','`).
- **No header row.** `dtype=float` turns any text header into `NaN` and corrupts the data — the first row must already be numbers.
- **Column 0 is time, monotonically increasing.** It is the integration/save grid (`t0 → t1`).
- **Rectangular.** Every row must have the same number of columns; ragged rows fail to parse.
- **No blank/missing cells.** Missing values become `NaN`, which makes the loss `NaN` (sanitized to `1e10`), so the fit cannot make progress. Encode unmeasured points with an explicit **mask column**, not blanks.
- **File present and named correctly** at `sessions/<session>/inputs/<FILENAME_DATA>`.
- **Column order is positional and must match `user_model.py`.** Columns 1..N (after time) are consumed in file order; the loss/writeout code indexes `dataset[:, k]` by position. `COLUMN_INFO` in the XML is NOT used to remap. A wrong column order silently fits the wrong data.

**XML:**
- **`N_TRAINABLE_PARAMETERS` must equal the number of `<PARAM>` blocks** in `TRAINABLE_PARAMETER_DESCRIPTION`.
- **Ordering is load-bearing.** Trainable-parameter order defines the optimization vector index order; integrated-variable order defines `y[]` indexing. Both must match the unpacking order in `user_model.py`. A mismatch does not raise — it silently fits the wrong quantity.
- **`STEPSIZE_RTOL` and `STEPSIZE_ATOL`** (and any `POP_STEPSIZE_*`) must each have exactly one comma-separated value **per integrated variable**.
- **Names must be globally unique** across trainable + fixed + integrated variables (`check_name_uniqueness` raises) **and valid Python identifiers** (no spaces/hyphens — they become dict keys/identifiers in generated code).
- **`LOGSCALE` must be exactly `Y` or `N`** (anything else raises).
- **Micro-format:** every leaf is `<P> KEY = VALUE </P>` with exactly one `=`. A value containing `=` breaks parsing.
- **Numeric fields must parse:** `MIN_VAL/MAX_VAL/VALUE/INIT_VAL` → float; `N_TRAINABLE_PARAMETERS/POPULATION_SIZE/NUM_ITERS/PROCESSORS/MAX_STEPS` → int.
- **Required settings (no safe default — missing → crash):** `STEPSIZE_RTOL`, `STEPSIZE_ATOL`, `MAX_STEPS`, both `NUM_ITERS` (population + gradient), `POPULATION_SIZE`, `PROCESSORS`, and each trainable parameter's `MIN_VAL`/`MAX_VAL`/`LOGSCALE`.
- **At least one `<EXPERIMENT>`** with a `FILENAME_DATA`.
- **`INITIAL_CONDITIONS` names must match.** Each `VAR/NAME` inside an `<INITIAL_CONDITIONS>` block must exactly match a name in `INTEGRATED_SYSTEM_DESCRIPTION` (otherwise `get_y0` raises).
- **Every integrated variable needs an `INIT_VAL`,** and the variable count must equal the ODE state dimension returned by `user_defined_system`.

### Non-critical — may degrade fit quality (no crash)

- **Initial conditions come from the XML, not the data.** The integrator starts from the XML `INIT_VAL`s; the data's first row is compared against the simulated initial state. If row 0 of the data disagrees with the XML ICs for an observed variable, that mismatch is a fixed penalty the optimizer cannot remove — make them consistent.
- **The loss is not normalized by the framework.** `_compute_loss_problem` should scale residuals (e.g. divide by per-column max) so the loss is roughly in [0, 1]; an unscaled loss can hurt gradient step sizes and convergence.
- **Search ranges and log-scaling.** Ranges that are far too wide/narrow, or not flagged `LOGSCALE = Y` when they span many orders of magnitude, slow or stall the global search.
- **`GRADIENT_OPT/SETTINGS` silently ignores unknown keys** (unlike other sections, which raise). A typo'd gradient setting is dropped, so a value you think you set may not be applied — double-check spelling.
- **`COLUMN_INFO` is parsed but unused.** Do not rely on it to remap columns; only the physical column order vs. the loss code matters.
- **Mismatched timescales across experiments.** Experiments may each have their own length/time grid, but if their magnitudes differ greatly the averaged loss can be dominated by a single experiment.
