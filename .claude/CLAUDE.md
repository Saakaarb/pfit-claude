# pfit-claude: ODE Parameter Fitting with Claude Code

## What This Project Does

Fits unknown parameters of a system of ODEs to user-provided time-series data using a two-stage optimization:
1. **PSO** (Particle Swarm Optimization) — global search
2. **Gradient-based** (L-BFGS / Adam via optax) — local refinement

The LLM layer (Claude Code skills) handles code generation and validation. The optimization pipeline runs as plain Python.

## Workflow

**Standard (user writes ODE):**
```
/pfit-skeleton   →   user fills ODE logic   →   /pfit-check   →   /pfit-jax   →   python fit_parameters.py <session_name>
```

**From paper (model extracts ODE):**
```
/pfit-from-source   →   /pfit-check   →   /pfit-jax   →   python fit_parameters.py <session_name>
```

Each session lives in `sessions/<session_name>/` with the structure:
```
sessions/<session_name>/
├── inputs/
│   ├── user_input.xml        ← USER PROVIDES: configuration
│   └── <data>.csv            ← USER PROVIDES: time-series data
├── generated/
│   ├── user_model.py         ← created by /pfit-skeleton, filled by user
│   ├── user_input_check.txt  ← created by /pfit-check
│   └── generated_script.py  ← created by /pfit-jax
└── outputs/                  ← created by fit_parameters.py
    ├── final_design_point.csv
    ├── result_solution_exp1.csv   ← one file per experiment
    ├── result_solution_exp2.csv   ← (only exp1 exists for single-experiment fits)
    ├── pso_fitting.log
    └── NODE_fitting.log
```

## Key Reference Files

| File | Purpose |
|------|---------|
| `lib/LLM/user_model_generation_instructions.txt` | Rules for generating user_model.py skeleton |
| `lib/LLM/user_file_check_instructions.txt` | Rules for validating XML + user_model.py |
| `lib/LLM/inputs_fix_instructions.txt` | Rules for auto-correcting errors |
| `lib/LLM/developer_instructions.txt` | Rules for generating JAX generated_script.py |
| `lib/utils/user_model_sample_unpopulated.py` | Template skeleton for user_model.py |
| `lib/utils/user_model_sample_populated.py` | Populated example (Robertson system) |
| `lib/utils/output_sample.py` | Template for generated_script.py (copy fixed functions verbatim) |
| `lib/utils/user_input_sample.xml` | Example XML (Robertson system) |
| `examples/` | Complete worked examples (vanderpol, robertson, ARC, piezo) |

## Multi-Experiment Execution Model

**This is critical to understand before generating or reviewing any code.**

Multiple `<EXPERIMENT>` blocks in the XML mean the same parameter set is fitted simultaneously against multiple datasets (e.g. the same ODE system measured under different initial conditions or different runs). The framework handles aggregation automatically:

- `_compute_loss_problem(constants, trainable_variables)` is called **once per experiment** with that experiment's own `constants` dict (its own `dataset`, `t_eval`, `init_cond`)
- The framework averages the scalar losses across all experiments — `_compute_loss_problem` must return a scalar for **one** experiment only; do NOT loop over experiments or aggregate inside this function
- `_write_problem_result(constants, trainable_variables)` is likewise called **once per experiment**; outputs are saved as `result_solution_exp1.csv`, `result_solution_exp2.csv`, etc.
- Each experiment's `constants["init_cond"]` already reflects that experiment's initial conditions (global defaults merged with any per-experiment `INITIAL_CONDITIONS` overrides from the XML)

The `dataset`, `t_eval`, and `init_cond` in `constants` always represent **one experiment at a time**. Never write code that tries to loop over or concatenate multiple datasets inside the user functions.

---

## XML Format

The user provides `inputs/user_input.xml`. Key sections:

**Single-experiment** (most common):
```xml
<FIT>
  <EXPERIMENT>
    <FILENAME><E> FILENAME_DATA = mydata.csv </E></FILENAME>
    <COLUMN_INFO>
      <COLUMN>
        <P> NAME = x1 </P>
        <P> COLUMN_INDEX = 1 </P>   <!-- 0-based index -->
      </COLUMN>
    </COLUMN_INFO>
  </EXPERIMENT>

  <MODEL>
    <TRAINABLE_PARAMETERS>
      <P> N_TRAINABLE_PARAMETERS = 2 </P>
    </TRAINABLE_PARAMETERS>
    <TRAINABLE_PARAMETER_DESCRIPTION>
      <PARAM>
        <P> PARAMETER_NAME = mu </P>
        <P> MIN_VAL = 0.001 </P>
        <P> MAX_VAL = 100.0 </P>
        <P> LOGSCALE = Y </P>   <!-- Y if range spans many orders of magnitude -->
      </PARAM>
    </TRAINABLE_PARAMETER_DESCRIPTION>
    <FIXED_PARAM_DESCRIPTION>
      <PARAM>
        <P> NAME = some_const </P>
        <P> VALUE = 5.0 </P>
      </PARAM>
    </FIXED_PARAM_DESCRIPTION>
    <INTEGRATED_SYSTEM_DESCRIPTION>
      <VAR>
        <P> NAME = x1 </P>
        <P> INIT_VAL = 1.0 </P>
      </VAR>
    </INTEGRATED_SYSTEM_DESCRIPTION>
  </MODEL>

  <POPULATION_OPT>
    <SETTINGS>
      <P> POPULATION_SIZE = 100 </P>
      <P> NUM_ITERS = 20 </P>
      <P> PROCESSORS = 8 </P>
      <P> ALGORITHM = PSO </P>   <!-- optional; PSO (default) or DE -->
    </SETTINGS>
  </POPULATION_OPT>

  <GRADIENT_OPT>
    <SETTINGS>
      <P> NUM_ITERS = 10 </P>
      <P> STEPSIZE_RTOL = 1e-7,1e-7 </P>
      <P> STEPSIZE_ATOL = 1e-9,1e-9 </P>
      <P> INITIAL_TIMESTEP = 1e-6 </P>
      <P> MAX_STEPS = 10000 </P>
      <P> INTEGRATOR = Kvaerno5 </P>   <!-- optional; default Kvaerno5. Stiff: Kvaerno3, Kvaerno5. Non-stiff: Dopri5, Dopri8, Tsit5 -->
      <P> GRADIENT_OPTIMIZER = lbfgs </P>   <!-- optional; lbfgs (default) or adam -->
      <P> INIT_VALUE_LR = 1e-4 </P>
      <P> END_VALUE_LR = 1e-5 </P>
      <P> TRANSITION_STEPS_LR = 2000 </P>
      <P> DECAY_RATE_LR = 0.9 </P>
    </SETTINGS>
  </GRADIENT_OPT>

  <PLOTTING_INFO>
    <P> WRITE_RESULTS = Y </P>
  </PLOTTING_INFO>
</FIT>
```

**Multi-experiment** — add one `<EXPERIMENT>` block per dataset; `<INITIAL_CONDITIONS>` overrides the global `INIT_VAL` for that experiment only. Variables not listed inherit the global initial condition:
```xml
<FIT>
  <EXPERIMENT>
    <FILENAME><E> FILENAME_DATA = run_A.csv </E></FILENAME>
    <INITIAL_CONDITIONS>
      <VAR>
        <P> NAME = x1 </P>
        <P> INIT_VAL = 1.0 </P>
      </VAR>
      <VAR>
        <P> NAME = x2 </P>
        <P> INIT_VAL = 0.0 </P>
      </VAR>
    </INITIAL_CONDITIONS>
  </EXPERIMENT>
  <EXPERIMENT>
    <FILENAME><E> FILENAME_DATA = run_B.csv </E></FILENAME>
    <INITIAL_CONDITIONS>
      <VAR>
        <P> NAME = x1 </P>
        <P> INIT_VAL = 2.0 </P>   <!-- different IC for this run -->
      </VAR>
    </INITIAL_CONDITIONS>
  </EXPERIMENT>
  <!-- MODEL, POPULATION_OPT, GRADIENT_OPT identical to single-experiment -->
</FIT>
```

Key rules for multi-experiment XML:
- Each `EXPERIMENT` must have its own `<FILENAME>` with a unique CSV file
- `<INITIAL_CONDITIONS>` inside an `EXPERIMENT` is optional; omit it if ICs are the same for all runs
- `VAR/NAME` inside `INITIAL_CONDITIONS` must exactly match a variable name in `INTEGRATED_SYSTEM_DESCRIPTION` — a mismatch causes a runtime crash
- Different experiments having different initial conditions is expected and valid — do NOT flag it as an error

## user_model.py Structure

Three functions the user implements in pseudocode (numpy-style, not JAX):

```python
def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    # trainable_parameters and fixed_parameters treated as dicts for user convenience
    # Must return list/array of derivatives for EVERY integrated variable

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    # Must return a scalar
    # Should be normalized so loss is likely between 0 and 1

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    # Returns array to be written to result_solution_expN.csv
```

**Multi-experiment note:** `dataset` and `t_eval` always represent a **single experiment's data**. The framework calls each function once per experiment and averages losses automatically. Do NOT write loops over multiple datasets inside these functions.

## generated_script.py Structure

JAX-jittable version generated by /pfit-jax. Fixed functions are copied verbatim from `lib/utils/output_sample.py`:
- `scale_value()` — maps parameter space → [-1, 1]
- `unscale_value()` — maps [-1, 1] → parameter space (handles log-scale)
- `_integrate_system()` — diffrax ODE solver (class set by `INTEGRATOR` in XML, default `Kvaerno5`)

User functions are translated to JAX:
- `user_defined_system()` — JAX ODE RHS
- `_compute_loss_problem()` — JAX loss, @jax.jit
- `_write_problem_result()` — NOT jitted (does file I/O)

**Critical**: `max_steps` in `_integrate_system` must be the literal integer from `GRADIENT_OPT/MAX_STEPS` in the XML.

**Multi-experiment**: `_compute_loss_problem` and `_write_problem_result` are each called once per experiment by the framework with a single `constants` dict. Do NOT add any cross-experiment aggregation logic inside these functions — the framework averages losses automatically.

## Parameter Scaling

All optimization is done in [-1, 1] space. For log-scale parameters, log10 is applied before linear scaling.

## Python Environment

Always use the venv in the project root for any Python commands:
```bash
/home/sbhatnagar/wsl/pfit-claude/venv/bin/python3
```
Never use the system `python` or `python3` directly — JAX and diffrax are only installed in this venv.

## Skills Available

- `/pfit-skeleton` — Generate user_model.py skeleton from XML (user provides XML first)
- `/pfit-from-source` — Generate both XML and user_model.py from a paper/reference in one step
- `/pfit-check` — Validate XML + user_model.py, auto-correct errors
- `/pfit-jax` — Convert user_model.py to JAX generated_script.py

---

## Stage Details

### `/pfit-skeleton` — Generate Skeleton

**Input:** `sessions/<session>/inputs/user_input.xml`
**Output:** `sessions/<session>/generated/user_model.py`

**Reference files to read:**
- `lib/LLM/user_model_generation_instructions.txt`
- `lib/utils/user_model_sample_unpopulated.py` — skeleton template
- `lib/utils/user_model_sample_populated.py` — Robertson example
- `lib/utils/user_input_sample.xml` — Robertson example XML

**What to generate:**
- Three function stubs: `user_defined_system`, `_compute_loss_problem`, `writeout_description`
- Trainable parameter names unpacked from `TRAINABLE_PARAMETER_DESCRIPTION` (with ordering comments)
- Fixed parameter names from `FIXED_PARAM_DESCRIPTION`
- Integrated variable names and initial conditions from `INTEGRATED_SYSTEM_DESCRIPTION`
- `import numpy as np` at the top
- All comments from the template preserved; no functions left empty

**Rules:**
- Do NOT leave any function empty — add stubs with comments showing what to fill in
- Do NOT add boilerplate text; the file must be pure Python
- Users treat `trainable_parameters` and `fixed_parameters` as dicts (pseudocode convenience)
- If the XML contains multiple `<EXPERIMENT>` blocks, add a comment near the top of each function noting that `dataset` and `t_eval` represent a single experiment's data — the framework calls the function once per experiment

---

### `/pfit-check` — Validate and Auto-Correct

**Input:** XML + `user_model.py`
**Output:** edited XML, edited `user_model.py`, report at `sessions/<session>/generated/user_input_check.txt`

**Reference files to read:**
- `lib/LLM/user_file_check_instructions.txt` — what to validate
- `lib/LLM/inputs_fix_instructions.txt` — how to auto-correct
- `lib/LLM/model_output_check_inputcheck_instructions.txt` — report format

**Loop:** validate → if critical errors → auto-correct → re-validate, up to 3 iterations.

#### XML checks

**EXPERIMENT blocks:**
- At least one must exist (critical if missing)
- Each must have `FILENAME_DATA` (critical if missing)
- `INITIAL_CONDITIONS/VAR/NAME` must match a variable in `INTEGRATED_SYSTEM_DESCRIPTION` (critical mismatch → remove that VAR block entirely, do not guess a corrected name)
- `INITIAL_CONDITIONS` is optional per experiment — absence means that experiment uses the global `INIT_VAL` values
- Different experiments having different initial conditions is expected and valid — do NOT flag it
- For multi-experiment fits: verify every referenced CSV file exists in `sessions/<session>/inputs/`

**TRAINABLE_PARAMETER_DESCRIPTION:**
- Search ranges must be sensible
- Parameters spanning many orders of magnitude must use `LOGSCALE = Y` — it is your job to flag this, not the user's
- All names must be valid Python identifiers
- No duplicate names across trainable, fixed, or integrated variables (flag as top-priority critical error)

**FIXED_PARAM_DESCRIPTION:** fixed values must be reasonable; names must be valid Python identifiers

**INTEGRATED_SYSTEM_DESCRIPTION:** variable names must be pythonic; initial values must be sensible

**POPULATION_OPT:**
- `POPULATION_SIZE < 20` → critical
- `POPULATION_SIZE > 1000` and no `POP_STEPSIZE_RTOL` → warning (likely very slow)
- `PROCESSORS > 8` → warning (hard limit in fit_parameters.py)
- `NUM_ITERS < 5` → warning
- `POP_STEPSIZE_RTOL` tighter than `STEPSIZE_RTOL` → warning (zero-order tolerances should be looser)
- Budget check: `POPULATION_SIZE × NUM_ITERS` vs. `20 × N_TRAINABLE_PARAMETERS²` — flag if insufficient

**GRADIENT_OPT:**
- `NUM_ITERS < 3` → warning
- `MAX_STEPS < 1000` → warning
- `INIT_VALUE_LR < END_VALUE_LR` → critical (inverted LR schedule)
- Any `STEPSIZE_RTOL` or `STEPSIZE_ATOL` < 1e-12 → warning

#### user_model.py checks

**`user_defined_system`:**
- Every integrated variable must have a derivative defined and returned (critical)
- No undefined parameters used
- Only numpy/math libraries — no scipy, torch, etc.
- Code must be convertible to JAX (critical)

**`_compute_loss_problem`:**
- Must return a scalar or 1D array of length 1 (critical)
- Loss must be normalized so it likely falls between 0 and 1 (critical)
- Only numpy/math libraries
- Must NOT loop over or aggregate multiple datasets — it handles one experiment at a time (critical if violated)

**`writeout_description`:**
- Must return an array
- No undefined parameters used
- Must NOT loop over multiple datasets — one experiment at a time

#### Report rules
- Only list items the user needs to act on — if a check passes, omit it
- Critical errors: will certainly cause JAX compilation, runtime, or integration failure
- Warnings: might cause failure or poor convergence
- End report with: `Number of critical errors: N`
- Do NOT flag indentation errors, missing JAX imports, or non-JAX syntax (code is pseudocode)
- Do NOT suggest the user convert code to JAX themselves

#### Auto-correction rules
- Make MINIMAL changes — as few edits as possible to fix as many critical errors as possible
- Do NOT change code logic unless fundamentally wrong and flagged in the report
- Do NOT change column index mappings
- If an `INITIAL_CONDITIONS/VAR` name doesn't match, remove that VAR block — do not invent a name
- If a FILENAME is missing, flag but do not invent a filename
- If not possible to fix an error, ignore it

---

### `/pfit-jax` — Generate JAX Script

**Input:** XML + `user_model.py`
**Output:** `sessions/<session>/generated/generated_script.py`

**Reference files to read:**
- `lib/LLM/developer_instructions.txt` — conversion rules
- `lib/utils/output_sample.py` — template with fixed functions to copy verbatim
- `sessions/<session>/inputs/user_input.xml` — for parameter info and `MAX_STEPS`

#### File structure (in this exact order)

```python
import jax
import jax.numpy as jnp
import diffrax
from diffrax import RESULTS

jax.config.update("jax_enable_x64", True)

@jax.jit
def scale_value(...):           # COPY VERBATIM from output_sample.py

@jax.jit
def unscale_value(...):         # COPY VERBATIM from output_sample.py

@jax.jit
def user_defined_system(t, y, other_args):   # TRANSLATE from user_model.py

@jax.jit
def _integrate_system(...):     # COPY from output_sample.py, fill max_steps + substitute SOLVER_CLASS

@jax.jit
def _compute_loss_problem(constants, trainable_variables):  # TRANSLATE from user_model.py

def _write_problem_result(constants, trainable_variables):  # TRANSLATE, NOT jitted
```

#### Critical translation rules

- `scale_value`, `unscale_value`, `_integrate_system`: copied VERBATIM — do NOT modify
- `max_steps` in `_integrate_system`: must be the **literal integer** from `GRADIENT_OPT/SETTINGS/MAX_STEPS` in the XML — not a variable, not a string, a literal number
- The solver line `diffrax.SOLVER_CLASS()` in `_integrate_system` must be replaced with `diffrax.<IntegratorName>()` where `IntegratorName` comes from `GRADIENT_OPT/SETTINGS/INTEGRATOR` in the XML; if absent, default to `Kvaerno5`. Valid options — stiff (implicit): `Kvaerno3`, `Kvaerno5`; non-stiff (explicit): `Dopri5`, `Dopri8`, `Tsit5`
- In `user_defined_system`: unpack trainable parameters via `unscale_value(trainable_variables, min_val, max_val, is_logscale)` in the XML order; use `fixed_parameters` as a dict
- In `_compute_loss_problem`: call `_integrate_system`, handle integration failure via `jnp.where(failed, constants["error_loss"], loss_value)`
- Trainable parameters in pseudocode are treated as dicts — translate these to **vector indexing** in JAX; the order is identical to the XML order (also documented in pseudocode comments)
- Fixed parameters remain a dict in JAX — do NOT treat as a vector
- Function signatures must exactly match those in `output_sample.py`
- `_write_problem_result` must NOT have `@jax.jit` (does file I/O)
- No boilerplate text in output — pure Python only, no markdown fences

#### Multi-experiment translation rules

- `_compute_loss_problem` receives `constants` for **one experiment** — `constants["dataset"]`, `constants["t_eval"]`, and `constants["init_cond"]` already contain that experiment's data and initial conditions
- Do NOT add loops, stacks, or concatenations across experiments inside `_compute_loss_problem` or `_write_problem_result` — the framework in `helper_functions.py` handles calling these functions per-experiment and averaging
- `_integrate_system` uses `constants["init_cond"]` as `y0` — for multi-experiment fits this is automatically set per-experiment by the framework before calling `_compute_loss_problem`
- The generated script is identical for single and multi-experiment fits; the per-experiment dispatch is entirely outside the generated script

#### SaveAt fix (critical)
Always use `diffrax.SaveAt(t0=True, ts=t_eval[1:])` — NOT `SaveAt(ts=t_eval)`. diffrax requires `saveat.ts` to be strictly monotone; when `t_eval[0] == t0`, using `ts=t_eval` triggers `_EquinoxRuntimeError: saveat.ts must be increasing or decreasing` because t0 is prepended implicitly. `t0=True, ts=t_eval[1:]` saves the initial point separately and keeps remaining ts strictly after t0.

#### After generating
Verify with:
```bash
cd sessions/<session_name> && /home/sbhatnagar/wsl/pfit-claude/venv/bin/python3 -c "import sys; sys.path.insert(0, '../..'); import generated.generated_script as gs; print('Import OK')"
```
If it fails, read the error, fix `generated_script.py`, and retry up to 3 times.
