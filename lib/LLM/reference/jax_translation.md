---
topic: Converting user_model.py pseudocode into the JAX generated_script.py
consumed_by: [pfit-jax]
generated: false
owns: >
  The generated-script structure, which functions are copied verbatim, the
  max_steps and solver substitutions, the integration-failure mask, the SaveAt
  convention, dict-to-vector translation, and the output formatting rules.
---

# JAX translation rules

Task: turn the three pseudocode functions in `user_model.py` into runnable,
JIT-compilable JAX equivalents with **identical logic**. The contract for those
functions (names, arguments, what they return) is in `user_model_contract.md`.

Reference template: `lib/utils/output_sample.py`. The input and output
signatures of the functions you produce MUST match it exactly.

**Never write a diffrax/jax/optax call from memory** — consult
`lib/LLM/api/diffrax.md` and `lib/LLM/api/jax.md`.

You MUST solve the problem: no function may be left empty.

## File structure, in this exact order

```python
import jax
import jax.numpy as jnp
import diffrax
from diffrax import RESULTS

jax.config.update("jax_enable_x64", True)

@jax.jit
def scale_value(...):            # COPIED VERBATIM
@jax.jit
def unscale_value(...):          # COPIED VERBATIM
@jax.jit
def user_defined_system(t, y, other_args):        # TRANSLATED
@jax.jit
def _integrate_system(constants, trainable_variables):   # COPIED, 2 substitutions
@jax.jit
def _compute_loss_problem(constants, trainable_variables):   # TRANSLATED
def _write_problem_result(constants, trainable_variables):   # TRANSLATED, NOT jitted
```

Those four imports are required. Add others only if genuinely needed.
`jax.config.update("jax_enable_x64", True)` must appear before any array is
created — without it every solve silently runs in float32.

## Copied verbatim — do not modify

`scale_value`, `unscale_value` and `_integrate_system` are copied identically
from `output_sample.py`, decorated with `@jax.jit`. `_integrate_system` has
exactly **two** permitted substitutions, described next.

### Substitution 1: `max_steps`

`max_steps` in `diffrax.diffeqsolve` must be the **literal integer** from
`GRADIENT_OPT/SETTINGS/MAX_STEPS` in the XML. Not a variable, not a dict lookup,
not left empty — a literal number. It is a compile-time (static) argument;
passing a traced value fails to compile.

### Substitution 2: the solver class

Replace the placeholder `diffrax.SOLVER_CLASS()` with `diffrax.<IntegratorName>()`
where `IntegratorName` comes from `GRADIENT_OPT/SETTINGS/INTEGRATOR` (default
`Kvaerno5` when absent). Like `max_steps`, this must be a literal class
instantiation, not a variable.

The valid names are exactly the solver table in `lib/LLM/api/diffrax.md`. A name
absent from that table does not exist in the pinned diffrax and will raise
`AttributeError`; the table also excludes solvers that exist but cannot work
here. If the XML names an invalid solver, stop and report it rather than
generating the script.

## Translation rules

- **Trainable parameters:** the pseudocode treats them as a dict; in JAX they are
  a **vector**. Unpack via
  `unscale_value(trainable_variables, min_val, max_val, is_logscale)` in the XML
  order. That order is fixed — never change it, never add or drop a parameter.
- **Fixed parameters stay a dict.** Do not treat them as a vector. They are used
  as given and are never trained or modified.
- **`_write_problem_result` must NOT be jitted** — it does file I/O.

### Integration-failure mask

In `_compute_loss_problem` the mask must be written exactly as:

```python
failed = jnp.invert(result == RESULTS.successful)
```

then applied as `jnp.where(failed, constants["error_loss"], loss_value)`.

Do NOT enumerate individual failure codes. diffrax defines 14 `RESULTS` codes
and only `successful` yields a usable trajectory; `dt_min_reached`,
`nonlinear_divergence`, `nonfinite` and `max_steps_rejected` all return
trajectories containing `inf`/`NaN`. Unmasked, those score as a genuine fit and
hand the gradient stage NaN gradients, stalling it with no error reported. Full
code table: `lib/LLM/api/diffrax.md`.

### SaveAt

Always `diffrax.SaveAt(t0=True, ts=t_eval[1:])` — never `SaveAt(ts=t_eval)`.
diffrax requires `saveat.ts` to be strictly monotone, and t0 is prepended
implicitly, so when `t_eval[0] == t0` the latter raises
`_EquinoxRuntimeError: saveat.ts must be increasing or decreasing`.

### NaN-safe arithmetic

When the loss masks NaNs (ragged sampling), sanitise **before** dividing —
`jnp.where(mask, dataset, 0.0)` first — so NaN never enters the autodiff graph.
See `staggered_data.md`.

## Output format

The response must be **pure Python**, usable as-is: no boilerplate prose, no
markdown fences such as ```` ```python ````, nothing but code and Python
comments.

## Performance note

If the first global-search iteration takes minutes, lower `MAX_STEPS`. During
global search the population contains wild parameter sets; for systems that can
blow up, those drive the stiff solver to grind to `max_steps` before failing, at
a cost of `population_size x max_steps x (stiff solve)`. Dropping `MAX_STEPS`
(e.g. 100000 -> 5000) makes doomed solves fail fast and return `error_loss`
while still resolving good members — often a 10-40x speedup with no loss of fit
quality. Halving the population helps proportionally. Change `MAX_STEPS` in
**both** the XML and the literal in `generated_script.py`.

## Verification

After generating, confirm the script imports:

```bash
cd sessions/<session_name> && ../../venv/bin/python3 -c \
  "import sys; sys.path.insert(0, '../..'); import generated.generated_script as gs; print('Import OK')"
```

On failure, read the error, fix the script, and retry (up to 3 attempts).
