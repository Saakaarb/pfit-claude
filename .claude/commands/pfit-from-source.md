Generate both user_input.xml and a fully-populated user_model.py in one step from an ODE system described in a paper, textbook, or other reference.

Use this instead of `/pfit-skeleton` when the user has a source that specifies
the ODE system directly. This file is PROCEDURE only — the formats and rules it
produces against live in the reference files below.

## Reference files (read before writing anything)

| File | Why |
|---|---|
| `lib/LLM/reference/xml_format.md` | the XML schema, defaults and valid values |
| `lib/LLM/reference/user_model_contract.md` | the three functions and what each must return |
| `lib/LLM/reference/input_constraints.md` | what the generated inputs must satisfy |
| `lib/LLM/api/diffrax.md` | valid `INTEGRATOR` names — never propose one from memory |
| `lib/LLM/api/optax.md` | valid gradient optimizers and their real defaults |
| `lib/LLM/reference/staggered_data.md` | if the source reports observables at different time points |

Worked example: `lib/utils/user_model_sample_populated.py`.

## Steps

### 1. Gather inputs

Ask for (if not given): the **session name**, the **reference** (URL or local
path), and the **data file(s)** already present in `sessions/<session>/inputs/`.

### 2. Read the reference

WebFetch for a URL, Read for a local path. Extract every ODE system, state
variable and parameter. If the source is inaccessible or unreadable, say so
immediately and ask the user to paste the equations.

### 3. Clarify the system (REQUIRED before writing anything)

Present your findings and wait for confirmation:

1. **Which system** — if the source has several models or variants, list them and
   ask which to fit.
2. **State variables** — names and physical meaning.
3. **Parameters** — all of them, with your proposed trainable/fixed split
   (trainable = treated as unknown or estimated; fixed = given an exact value).
4. **Initial conditions** — the values you found; ask for confirmation.
5. **Multi-experiment** — ask explicitly whether they have several datasets to
   fit simultaneously. If yes: how many, which CSV each is, and whether the
   initial conditions differ per run.

### 4. Clarify the data

Read the first 20 rows of each CSV. Report per file: column count, headers or
index labels, sample values, and your best guess at what each column is. Ask the
user to confirm the column mapping (which index is time; which column maps to
which state variable; whether any column is a derived observable). Assume one
layout for all files unless told otherwise. Wait for the answer.

### 5. Propose settings

Present defaults and let the user override:

- **Search ranges:** if the source gives a value `v`, propose `[v/100, v*100]`.
  If it gives none, propose a physically reasonable range and flag it explicitly
  for review.
- **Population:** `POPULATION_SIZE = max(50, 10 x N_TRAINABLE)`, `NUM_ITERS = 20`,
  `PROCESSORS = 4`.
- **Gradient:** `NUM_ITERS = 10`, `MAX_STEPS = 10000`,
  `INITIAL_TIMESTEP = 1e-6` (adjust if the problem's timescale is far from 1),
  `INIT_VALUE_LR = 1e-4`, `END_VALUE_LR = 1e-5`,
  `TRANSITION_STEPS_LR = 2000`, `DECAY_RATE_LR = 0.9`.
- **Tolerances:** one `STEPSIZE_RTOL`/`STEPSIZE_ATOL` value per integrated
  variable.
- **Solver / optimizer:** propose only values valid per the API digests.

Wait for the answer.

### 6. Write user_input.xml

Per `xml_format.md`, to `sessions/<session>/inputs/user_input.xml`.

### 7. Write user_model.py

A **fully-populated** model per `user_model_contract.md`, implementing the RHS
exactly as the source specifies. Write to
`sessions/<session>/generated/user_model.py`, creating `generated/` if needed.

### 8. Report

Tell the user what was generated (parameters, variables, equation summary) and
every assumption you made (guessed ranges, inferred ICs). Then: "Review both
files, then run `/pfit-check <session>` to validate before JAX generation."

## Rules

- Write NO files until the clarification rounds (steps 3-5) are complete.
- The data CSV is always user-provided — never generate or modify it.
