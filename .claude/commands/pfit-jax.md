Convert user_model.py to a JAX-jittable generated_script.py for the current session.

## Steps

1. Ask the user for the session name if not provided as an argument. The session directory is `sessions/<session_name>/`.

2. Read the following files:
   - `sessions/<session_name>/inputs/user_input.xml` — for parameter info and MAX_STEPS value
   - `sessions/<session_name>/generated/user_model.py` — user pseudocode to convert
   - `lib/LLM/developer_instructions.txt` — JAX conversion rules
   - `lib/utils/output_sample.py` — template with fixed functions to copy verbatim

3. Following `developer_instructions.txt`, generate `generated_script.py`:

   **File structure** (in this exact order):
   ```python
   import jax
   import jax.numpy as jnp
   import diffrax
   from diffrax import RESULTS

   jax.config.update("jax_enable_x64", True)

   @jax.jit
   def scale_value(...):       # COPY VERBATIM from output_sample.py

   @jax.jit
   def unscale_value(...):     # COPY VERBATIM from output_sample.py

   @jax.jit
   def user_defined_system(t, y, other_args):   # TRANSLATE from user_model.py

   @jax.jit
   def _integrate_system(...): # COPY VERBATIM from output_sample.py, but fill max_steps

   @jax.jit
   def _compute_loss_problem(constants, trainable_variables):  # TRANSLATE from user_model.py

   def _write_problem_result(constants, trainable_variables):  # TRANSLATE, NOT jitted
   ```

4. **Critical translation rules:**
   - `scale_value`, `unscale_value`, `_integrate_system` must be copied VERBATIM from `output_sample.py` — do NOT modify them
   - `max_steps` in `_integrate_system` must be the literal integer from `MAX_STEPS` in `GRADIENT_OPT/SETTINGS` of the XML — not a variable, a literal number
   - In `user_defined_system`: unpack trainable parameters via `unscale_value(trainable_variables, min_val, max_val, is_logscale)` in the same order as the XML; use `fixed_parameters` as a dict
   - In `_compute_loss_problem`: call `_integrate_system`, handle failure via `jnp.where(failed, constants["error_loss"], loss_value)`
   - trainable_parameters in user pseudocode are treated as dicts — translate these to vector indexing in JAX
   - fixed_parameters remain a dict in JAX
   - The function signatures must exactly match those in `output_sample.py`
   - No boilerplate text in the output — pure Python only, no markdown code fences

5. Write the result to `sessions/<session_name>/generated/generated_script.py`.

6. **Verify** the generated script by running:
   ```bash
   cd sessions/<session_name> && python -c "import sys; sys.path.insert(0, '../..'); import generated.generated_script as gs; print('Import OK')"
   ```
   If this fails with a syntax or import error, read the error, fix `generated_script.py`, and retry. Repeat up to 3 times.

7. Tell the user:
   - If successful: "generated_script.py created and verified. Run `python fit_parameters.py <session_name>` to start optimization."
   - If import errors remain after retries: show the error and the relevant section of generated_script.py so the user can investigate.

## Key Reminders
- `_write_problem_result` must NOT have `@jax.jit`
- The order of trainable parameters in the JAX vector must match the XML order exactly
- `jax.config.update("jax_enable_x64", True)` must appear near the top
- Do not include any imports beyond what is needed
- **SaveAt fix**: Always use `diffrax.SaveAt(t0=True, ts=t_eval[1:])` — NOT `SaveAt(ts=t_eval)`. diffrax (via equinox) requires `saveat.ts` to be strictly monotone; when `t_eval[0] == t0`, using `ts=t_eval` causes an `_EquinoxRuntimeError: saveat.ts must be increasing or decreasing` at runtime because t0 is implicitly prepended to the save list. Using `t0=True, ts=t_eval[1:]` tells diffrax to save the initial point separately and keeps the remaining ts strictly after t0.
- **Multi-experiment**: the generated script is identical whether there is one experiment or many. `_compute_loss_problem` and `_write_problem_result` each handle one experiment's `constants` dict. The framework in `helper_functions.py` calls them once per experiment and averages the losses. Do NOT add any experiment loops, stacks, or aggregations inside the generated functions — this is a critical error that will cause tracing failures or silently wrong results.
