Start a new fitting session: produce user_input.yaml and a populated user_model.py from a description of the ODE system, with or without a source document.

**This is the only entry point for a new problem.**

 The config's `trainable_parameters` and `integrated_variables` orderings are load-bearing and
are properties of the equations, so the config and the model must be written
**together, by this skill**, from the same understanding. Writing one before the
other is what makes an ordering mismatch possible, and a mismatch does not raise
— it silently fits the wrong quantity.

This file is PROCEDURE only. The formats and rules it produces against live in
the reference files below.

## Reference files (read before writing anything)

| File | Why |
|---|---|
| `lib/LLM/reference/cold_start.md` | the invariant: setup choices are made without the solution |
| `lib/LLM/reference/yaml_format.md` | the config schema, defaults and valid values |
| `lib/LLM/reference/user_model_contract.md` | the three functions and what each must return |
| `lib/LLM/reference/input_constraints.md` | what the generated inputs must satisfy |
| `docs/ui_input_design.md` | prose-first intake, the full mental picture completeness gate, and targeted follow-up prompts |
| `lib/LLM/reference/tuning_rules.md` | R7: choosing between Adam and L-BFGS |
| `lib/LLM/api/diffrax.md` | valid `integrator` names — never propose one from memory |
| `lib/LLM/api/optax.md` | valid gradient optimizers and their real defaults |
| `lib/LLM/reference/staggered_data.md` | if observables are reported at different time points |

Worked example: `lib/utils/user_model_sample_populated.py`.

## The five required inputs

A session cannot be specified without all five. A paper is likely to
carry all of them. Establish which are present **before** starting the interview,
and handle each absence by its own rule:

| # | Input | If missing |
|---|---|---|
| 1 | **The equation system** | **Fatal.** Nothing can substitute for it. Say so and stop; ask the user to supply the equations in any form. |
| 2 | **The loss formulation** | **Propose one** from the declared dataset columns (see below), and have the user confirm it. |
| 3 | **The parameter ranges** | If the source publishes values, use them: `[v/100, v*100]`, four decades centred on the published value. Otherwise the **user must supply them** — do not invent ranges and proceed. |
| 4 | **The dataset** | **Fatal.** Stop and ask for the CSV(s). Never generate data. If the data can be sampled from a provided document figure, warn the user that this can be inaccurate. |
| 5 | **The initial conditions** | The **user must supply them**, for every state. They are never fitted and cannot be read off the data, so a guess here is a permanent error in the fit. Where a state is unobserved, say plainly that its value is an assumption. |

## Interview style

Prefer a prose-first workflow. The user may supply a paper, a writeup, pasted
equations, and CSV files; they should not be asked to hand-author framework
tables, YAML, or `user_model.py`.

Extract the structured study spec yourself: equations, states, parameter split,
initial conditions, dataset column roles, forcing sources, derived observables,
and loss. Then perform a completeness check.

Frame completeness as a full mental picture of the run. Before writing files,
you must be able to describe, end to end, what the solver integrates for each
experiment, which experiment-specific values differ, how each CSV column enters
the RHS or loss, what each fitted parameter can change, and what artifact will
be written for plotting. Estimate that picture from the initial source, then ask
only for the pieces still missing from it.

If the supplied information is complete and unambiguous, do not ask extra
questions before creating the session. Show a short review of the inferred
study spec and proceed to write the files.

If something is missing or ambiguous, ask only targeted questions for the
unresolved items. Do not run a fixed interview checklist when the answer is
already present in the source. Good questions name the exact unresolved symbol,
column, or setting, for example:

- "The equations use `dose_rate(t)`, and the CSV has
  `dose_rate_uM_per_min`. Should I use that column as the time-varying input?"
- "The CSV has `fluorescence_sd`. Should this be the uncertainty for
  `fluorescence` in the loss?"
- "`baseline` appears in the observable formula but has no bounds. Is it fitted
  or fixed?"

The review surface should be short and change-oriented: show what will be used
for equations, fitted parameters, initial conditions, column roles, forcing
inputs, observables, and loss. Prefer accept/change phrasing over asking the
user to edit a table.

### Proposing a loss (input 2)

Only possible once every dataset column has a declared meaning, so do the column
declaration first. Then:

- **Uncertainty columns present** (`uncertainty_of`) — propose a sigma-weighted
  residual, `mean(((model - measured) / sigma)**2)`, using only finite
  measurements with finite positive sigma values. Error bars supplied and then
  ignored is a silent loss of information.
- **No uncertainty columns** — propose a per-column normalised RMSE, dividing
  each column's residual by that column's own `max|data|`. State the column
  magnitudes you measured as the reason: columns of unequal scale otherwise
  contribute unequally, and a small-magnitude observable becomes invisible.
- **NaN present in the data** — the reduction must be nan-safe
  (`staggered_data.md`), and say why.

Inspect the proposed observable and loss arithmetic for every division, not
just uncertainty weighting. For each denominator, decide whether it is
guaranteed finite and nonzero from the equations, data declarations, or
parameter bounds. If the denominator is data-derived, build the validity mask so
the denominator is finite and nonzero, replace invalid entries before dividing,
and exclude those residuals. If the denominator is model-derived and can cross
zero, ask the user how that case should be handled; adding an epsilon or other
regularisation changes the scientific objective and must be agreed to.

Present the proposal with its arithmetic and wait for confirmation. Never write a
loss the user has not agreed to.

## Steps

### 1. Gather inputs

Ask for (if not given): the **session name**, the **data file(s)** already
present in `sessions/<session>/inputs/`, and **a source, if one exists** — a URL,
a local path, pasted equations, or nothing at all.

Check the five required inputs above and report which are missing before going
further.

### 2. Establish the equations

- **With a source:** WebFetch a URL, Read a local path. Extract every ODE system,
  state variable and parameter. If the source is inaccessible or unreadable, say
  so immediately and fall through to the no-source path rather than guessing.
- **Without a source:** ask the user to state the system in whatever form is
  natural — LaTeX, plain text, or prose. Transcribe it back to them as an
  explicit list of equations and say what you assumed. Do not invent terms the
  user did not state; ask instead.

### 3. Clarify the system (REQUIRED before writing anything)

Resolve the items below from the source/writeup first. If they are complete and
unambiguous, include them in the short review and continue. If any item is
missing or ambiguous, ask only about that item:

1. **Which system** — if the source has several models or variants, list them and
   ask which to fit.
2. **State variables** — names, order, and physical meaning. Say explicitly that
   this order fixes `y[]` indexing.
3. **Parameters** — all of them, with your proposed trainable/fixed split
   (trainable = treated as unknown or estimated; fixed = given an exact value).
4. **Initial conditions** — the values you found or inferred; ask for
   confirmation. State which are measured and which are assumed.
5. **Multi-experiment** — ask explicitly whether they have several datasets to
   fit simultaneously. If yes: how many, which CSV each is, and whether the
   initial conditions differ per run.

### 4. Declare what every dataset column is (REQUIRED)

The CSV is a bare numeric matrix. Nothing in it says what column 2 means, and no
amount of later validation can recover that — so this must be established here,
and it is what makes the loss proposal possible at all.

Read the first 20 rows of each CSV. **If the file has a header row, it is the
user's own statement of the column meanings** — use it as the starting point
rather than guessing. A header is optional and skipped on load; the file may
equally well have none.

Infer each column's role from the header, source/writeup, equation symbols, and
loss description. Report per file only the concise inferred mapping: time,
fitted measurements, forcing inputs, uncertainty columns, and ignored columns.
Ask follow-up questions only for columns whose role or target is unresolved.
Then record the result as the `columns` block of every experiment (schema in
`yaml_format.md`):

- entry `i` describes column `i`, so the first entry is always time;
- `observes: <name>` on every measurement column, naming the model quantity it
  measures. That is either an integrated variable (measured directly, which also
  lets `/pfit-check` compare it against `init_val`) or a **declared observable**;
- a column measuring something the model computes rather than integrates — an
  open probability, a relative percentage, a prevalence — needs an entry in
  `model.observables` and a matching key from `_observables` in `user_model.py`.
  The column's own `name` may differ from the observable's: `name` is what the
  user calls it, `observes` is what the model calls it, and the `columns` block
  is the dictionary between the two;
- `uncertainty_of: <column>` for a standard-deviation column, which also
  determines the loss shape in input 2;
- For multi-dataset runs, the framework files used for fitting MUST use the
  same column layout: time first, then the same forcing/measurement/uncertainty
  columns in the same order. Raw user uploads may arrive
  with different names, order, extra columns, row counts, or sampling times. Do
  not edit those files or write modified copies. Instead, infer the mapping and
  show the proposed canonical fitting layout for the user to create.
- If a measured observable is absent from one experiment but present in another,
  ask the user to preserve the common layout and write missing values as
  blank/NaN in their data file, then use the nan-safe loss rules in
  `staggered_data.md`.
- Compare each experiment's ordered `observes` mappings and the measurement
  targets of `uncertainty_of`, and verify each declaration against its canonical
  CSV. Matching column counts or names alone is insufficient; the dataset
  checker currently checks only column counts across experiments. Row counts,
  sampling times, initial conditions, and forcing histories may differ.

Include the canonical layout in the short review before writing files. If a raw
file cannot be mapped into that layout without changing the scientific meaning
of the data, stop and ask the user for a corrected file. A wrong column map silently fits
the wrong data, and no downstream check can establish the actual scientific
meaning of a column.

### 5. Elicit the search ranges — by magnitude, not by min/max

This is the only input that needs real judgement from the user, so do not ask for
`min_val`/`max_val` directly. For each trainable parameter ask for its **units
and one typical value**, then propose:

- source gives a value `v` -> `[v/100, v*100]`;
- user gives a typical value `v` -> the same, stated as a proposal;
- neither -> derive a range from the data's own timescale (a rate the data can
  resolve at all is O(1/t_span) to O(1/dt_min)) and **flag it explicitly** as
  derived rather than known.

Set `logscale: true` whenever the proposed range spans more than two decades, and
say why. Never ask the user to decide log-scaling themselves.

### 6. Propose the control settings

Every control setting stays in the config and stays tunable — the user simply
does not have to type the first draft. Include these in the short review and let
the user override:

First choose `gradient_optimizer` using R7 in `tuning_rules.md`: default to
`adam`. Recommend `lbfgs` as an alternative when the loss is smooth and
well-normalised and the model is expected to match the data well.
Always write the selected `gradient_optimizer` explicitly in the config.
State the proposed choice and the model/data evidence for it, even when choosing
the default, and present only that optimizer's settings below. Use the starting
values below for this setup proposal; `/pfit-check` can recommend adjustments
with evidence.

- **Population:** `population_size = max(50, 10 x N_TRAINABLE)`, `num_iters = 20`,
  `processors = 4`.
- **Gradient (Adam):** start with `num_iters = 1000`,
  `init_value_lr = 5e-3`, and `transition_steps_lr = 100`.
- **Gradient (L-BFGS):** start with `num_iters = 50`. Learning-rate schedule
  settings (`init_value_lr`, `end_value_lr`, `transition_steps_lr`,
  `decay_rate_lr`) are not relevant; do not propose them as L-BFGS controls.
- **Integration (either optimizer):** `max_steps = 10000`,
  `initial_timestep = 1e-6` (adjust if the problem's timescale is far from 1).
- **Tolerances:** one `stepsize_rtol`/`stepsize_atol` value per integrated
  variable.
- **Solver / optimizer:** propose only values valid per the API digests.
- **Output:** set `output.write_results: true`; `/pfit-run` must save and
  inspect measured-versus-fitted plots after optimization.

Say that `/pfit-check` will re-derive these from the equations and the data with
evidence attached, so these are a starting point, not a commitment.

### 7. Write both files together

Per `yaml_format.md`, write `sessions/<session>/inputs/user_input.yaml`. Per
`user_model_contract.md`, write a **fully-populated**
`sessions/<session>/generated/user_model.py` implementing the RHS exactly as
established in step 2, creating `generated/` if needed.

Before reporting, cross-check the pair: the trainable-parameter order in the
config must match the unpacking order in the model, and the integrated-variable
order must match the `y[]` indexing and the returned derivative order.

### 8. Report

Tell the user what was generated (parameters, variables, equation summary) and
every assumption you made (guessed ranges, inferred ICs, any term you had to
supply). Then: "Review both files, then run `/pfit-check <session>` to validate
before JAX generation."

## Rules

- Write NO files until the clarification rounds (steps 3-6) are complete.
- Data CSVs are always user-provided — never modify them and never write
  derived data files. You may describe the canonical layout the framework needs,
  but the user must create or edit the CSVs.
- Do not perform unit conversions while canonicalizing CSVs unless the user
  explicitly asks for a conversion and gives the conversion rule.
- Never require the user to write or edit YAML by hand.
