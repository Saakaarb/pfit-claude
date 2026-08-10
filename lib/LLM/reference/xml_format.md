---
topic: The user_input.xml format — every section, field, default and valid value
consumed_by: [pfit-skeleton, pfit-from-source, pfit-check, pfit-jax]
generated: false
owns: >
  The XML schema, which fields are required vs optional, their defaults, and
  where the valid values for each come from.
---

# user_input.xml format

Parsed by `lib/utils/xmlread.py`. Every leaf is `<P> KEY = VALUE </P>` with
exactly one `=` (the `FILENAME` block uses `<E>` instead of `<P>`).

## Single experiment (most common)

```xml
<FIT>
  <EXPERIMENT>
    <FILENAME><E> FILENAME_DATA = mydata.csv </E></FILENAME>
    <COLUMN_INFO>
      <COLUMN>
        <P> NAME = x1 </P>
        <P> COLUMN_INDEX = 1 </P>   <!-- 0-based; parsed but NOT used to remap -->
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
        <P> LOGSCALE = Y </P>
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
      <P> ALGORITHM = PSO </P>
      <P> RANDOM_SEED = 0 </P>
      <P> POP_STEPSIZE_RTOL = 1e-5,1e-5 </P>
      <P> POP_STEPSIZE_ATOL = 1e-7,1e-7 </P>
    </SETTINGS>
  </POPULATION_OPT>

  <GRADIENT_OPT>
    <SETTINGS>
      <P> NUM_ITERS = 10 </P>
      <P> STEPSIZE_RTOL = 1e-7,1e-7 </P>
      <P> STEPSIZE_ATOL = 1e-9,1e-9 </P>
      <P> INITIAL_TIMESTEP = 1e-6 </P>
      <P> MAX_STEPS = 10000 </P>
      <P> INTEGRATOR = Kvaerno5 </P>
      <P> GRADIENT_OPTIMIZER = lbfgs </P>
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

## Multi-experiment

Add one `<EXPERIMENT>` block per dataset. `<INITIAL_CONDITIONS>` overrides the
global `INIT_VAL` for that experiment only; variables not listed inherit the
global value. See `project_context.md` for what multiple experiments *mean*.

```xml
<FIT>
  <EXPERIMENT>
    <FILENAME><E> FILENAME_DATA = run_A.csv </E></FILENAME>
    <INITIAL_CONDITIONS>
      <VAR>
        <P> NAME = x1 </P>
        <P> INIT_VAL = 1.0 </P>
      </VAR>
    </INITIAL_CONDITIONS>
  </EXPERIMENT>
  <EXPERIMENT>
    <FILENAME><E> FILENAME_DATA = run_B.csv </E></FILENAME>
    <INITIAL_CONDITIONS>
      <VAR>
        <P> NAME = x1 </P>
        <P> INIT_VAL = 2.0 </P>
      </VAR>
    </INITIAL_CONDITIONS>
  </EXPERIMENT>
  <!-- MODEL, POPULATION_OPT, GRADIENT_OPT identical to single-experiment -->
</FIT>
```

Rules specific to multi-experiment XML:

- Each `EXPERIMENT` needs its own `<FILENAME>` with a distinct CSV.
- `<INITIAL_CONDITIONS>` is optional per experiment; omit the block entirely when
  that experiment uses the global ICs. Do not emit an empty block.
- Different experiments having different initial conditions is expected and
  valid.

## Optional PATH section

```xml
<PATH>
  <P> USER_INPUT_DIR = inputs </P>
  <P> GENERATED_DIR = generated </P>
  <P> OUTPUT_DIR = outputs </P>
</PATH>
```

## Field reference

### MODEL

| Field | Required | Notes |
|---|---|---|
| `N_TRAINABLE_PARAMETERS` | yes | Must equal the number of `<PARAM>` blocks in `TRAINABLE_PARAMETER_DESCRIPTION` |
| `PARAMETER_NAME`, `MIN_VAL`, `MAX_VAL`, `LOGSCALE` | yes, per param | `LOGSCALE` must be exactly `Y` or `N`; anything else raises |
| `NAME`, `VALUE` (fixed) | per fixed param | |
| `NAME`, `INIT_VAL` (integrated) | per variable | |

Any unrecognised key inside a `<PARAM>` block raises.

### POPULATION_OPT/SETTINGS

| Field | Required | Default | Valid values |
|---|---|---|---|
| `POPULATION_SIZE` | yes | — | int |
| `NUM_ITERS` | yes | — | int |
| `PROCESSORS` | yes | — | int (XLA host device count) |
| `ALGORITHM` | no | `PSO` | `PSO`, `DE` |
| `RANDOM_SEED` | no | unset | int |
| `POP_STEPSIZE_RTOL` / `_ATOL` | no | falls back to the gradient tolerances | one comma-separated value per integrated variable |

**Any unknown key in this section raises a `ValueError`.**

### GRADIENT_OPT/SETTINGS

| Field | Required | Default | Valid values |
|---|---|---|---|
| `NUM_ITERS` | yes | — | int |
| `STEPSIZE_RTOL` / `STEPSIZE_ATOL` | yes | — | one comma-separated value per integrated variable |
| `INITIAL_TIMESTEP` | yes | — | float |
| `MAX_STEPS` | yes | — | int |
| `INTEGRATOR` | no | `Kvaerno5` | **see `lib/LLM/api/diffrax.md`** — the generated solver table is the only authority |
| `GRADIENT_OPTIMIZER` | no | `lbfgs` | `lbfgs`, `adam` (case-insensitive; see `lib/LLM/api/optax.md` for their real signatures) |
| `INITIAL_TIME` | no | `t_eval[0]` | float |
| `INIT_VALUE_LR`, `END_VALUE_LR`, `TRANSITION_STEPS_LR`, `DECAY_RATE_LR` | no | — | float; consumed by `adam` only |

**Unknown keys in this section are silently ignored** (unlike every other
section, which raises). A typo'd gradient setting is therefore dropped without
warning — check spelling.

### PLOTTING_INFO

| Field | Default | Valid values |
|---|---|---|
| `WRITE_RESULTS` | `N` | `Y`, `N` — `Y` writes `result_solution_expN.csv` |
