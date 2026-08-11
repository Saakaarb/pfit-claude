---
topic: How to derive setting recommendations from the model and data, before any fit has run
consumed_by: [pfit-check]
generated: false
owns: >
  The evidence-to-recommendation rules for cold-start settings, the confidence
  policy, the Recommendations report entry format, and the apply-on-confirmation
  policy shared with /pfit-diagnose.
---

# Tuning rules (cold start)

Applied by `/pfit-check` to produce the **Recommendations** section of its
report, *before* any fit has run. Post-fit tuning is a different job with
different evidence — see `diagnosis_rules.md`.

Severity thresholds on the numbers already in the XML belong to
`validation_rules.md`; this file owns recommendations **derived from the model
and the data**, i.e. from evidence the XML does not contain. The human-facing
catalogue of what is tunable at all is `docs/tunable_choices.md`.

## The rule that governs every recommendation

**Every recommendation must cite the evidence that produced it.** A suggestion
with no evidence line is noise — omit it and let the default stand. Evidence
means a specific thing you read: a line of the RHS, a column range in the CSV,
a parameter's `MIN_VAL`/`MAX_VAL` ratio, a count. Never recommend a value
because it is "typical".

If the current XML value is already reasonable, say nothing about that field.

## Evidence to gather first

Read these before writing any recommendation:

| Evidence | From | Used for |
|---|---|---|
| Whether the RHS is smooth | `user_defined_system` in `user_model.py` — look for `sign`, `abs`, `floor`, `ceil`, `clip`, `minimum`/`maximum`, `where` on a state-dependent condition | integrator family |
| Stiffness indicators | ratio of the largest to smallest rate/timescale constant appearing in the RHS, whether the RHS mixes fast and slow variables, whether any bound in the XML spans >4 decades | integrator family, tolerances |
| Time-span vs. sampling | first/last value and spacing of column 0 of each CSV; whether spacing is uniform or geometric | `MAX_STEPS`, `INITIAL_TIMESTEP` |
| Per-column magnitudes | max `|value|` of each data column | loss normalisation and weighting |
| Blank/NaN cells | any empty cell in a CSV | staggered-data handling |
| Bound width per parameter | `MAX_VAL / MIN_VAL` per trainable parameter | `LOGSCALE` |
| N, number of trainable parameters | XML | search budget, identifiability warning |
| Number of experiments | count of `<EXPERIMENT>` blocks | aggregation caveat |

## Rules

### R1 — Integrator family (highest value; always emit a verdict)

This is the only recommendation to make even when the XML already names a
solver, because a wrong family makes every other setting irrelevant.

Recommend a **family**, then name a specific solver only by reading it out of
the solver table in `lib/LLM/api/diffrax.md` — its `Kind` column gives the
family and its `Validated` column says which are exercised by this framework's
examples. Prefer a validated one. Never name a solver from memory, and never
recommend one absent from that table.

- Non-smooth RHS (any of the discontinuity indicators above) → recommend an
  **explicit** solver. Cite the exact offending expression. Explain that an
  implicit inner solve cannot converge across the discontinuity for *any*
  `MAX_STEPS`, so failures there are not fixable by raising it.
- Smooth RHS with a wide timescale ratio, or stiffness unknown → recommend an
  **implicit** solver. Note the asymmetry: implicit on a non-stiff smooth system
  costs ~3–10× per step but still converges.
- Smooth RHS, clearly non-stiff, and runtime matters → an explicit solver is
  reasonable, but say that this is a speed optimisation, not a correctness one.

### R2 — `LOGSCALE` per parameter

`MAX_VAL / MIN_VAL` >= 100 → recommend `LOGSCALE = Y`, quoting the ratio.
Below that, linear is fine. State it per parameter by name; do not issue a
blanket recommendation for all of them.

### R3 — Bounds

- A bound that is not strictly positive on a parameter marked `LOGSCALE = Y` is
  a critical error, not a recommendation — hand it to `validation_rules.md`.
- Bounds spanning more than ~8 decades → recommend narrowing using whatever the
  source or the data implies, and cite what that is. A 10-decade box wastes most
  of the population budget.
- If the source document (for a `/pfit-from-source` session) gives a literature
  value, recommend a box bracketing it by 1–2 decades rather than an arbitrary
  one.

### R4 — Two-tier tolerances

If `POP_STEPSIZE_RTOL`/`_ATOL` are absent, recommend setting them ~100× looser
than `STEPSIZE_RTOL`/`_ATOL`, and say what that buys: the global search does
`POPULATION_SIZE × NUM_ITERS` solves and does not need refinement accuracy.
Include the concrete values for this XML, one per state variable.

Do not recommend this when the loss is dominated by a fast transient that loose
tolerances would smear — if the CSV's early time spacing is orders of magnitude
finer than its late spacing, say so and recommend only ~10× looser.

### R5 — Search budget

Recommend `POPULATION_SIZE` >= 10·N and `NUM_ITERS` >= 20, N = number of
trainable parameters. Report the implied number of ODE solves
(`POPULATION_SIZE × NUM_ITERS`), and — if a per-solve time is known from a
previous session — the implied wall-clock at the current `PROCESSORS`. Budget is
the choice the user is best placed to make, so give them the arithmetic rather
than a bare number.

### R6 — Global algorithm

Default `PSO` needs no comment. Recommend `DE` when the evidence says the
landscape is rugged: a non-smooth RHS (R1 triggered), or a loss with masked NaN
regions. Say it is a robustness/speed trade, not a correctness one.

### R7 — Gradient optimizer

- `lbfgs` (default) for a smooth, well-normalised loss. Say nothing.
- Recommend `adam` when gradients are expected to be noisy: NaN-masked data,
  non-smooth RHS, or loose refinement tolerances. If recommending `adam`, you
  must also recommend `NUM_ITERS` >= 200 and an LR schedule (start ~1e-3, end
  ~1e-5) in the same entry — `adam` with L-BFGS-sized iteration counts does
  nothing.
- Never recommend the LR fields while `GRADIENT_OPTIMIZER` is `lbfgs`; they are
  silently ignored.

### R8 — Loss shape

The loss lives in `user_model.py`, so these are recommendations about code, not
XML. Quote the line you would change.

- Column magnitudes differing by more than ~10× and no per-column scaling in
  `_compute_loss_problem` → recommend the per-column-scaled RMSE pattern from
  `user_model_contract.md`, citing the two column ranges. Without it the largest
  column silently owns the fit.
- Any blank cell in a CSV and no NaN handling in the loss → recommend the
  NaN-safe pattern from `staggered_data.md`. Note that sanitising *after* any
  arithmetic still produces NaN gradients.
- More than one `<EXPERIMENT>` and datasets of visibly different quality or
  length → note that aggregation is an unweighted mean over experiments with no
  weighting available, so the user should decide whether that is acceptable
  (per-experiment weighting would require a code change).

### R9 — Identifiability, in advance

N > number of observed columns × 3, or two parameters that appear only as a
product/ratio in the RHS → note that these will likely show up as
non-identifiable directions in the post-fit sloppiness report, and that the fix
is fewer trainable parameters or a reparameterisation, not more iterations.
Cite the RHS expression where the parameters are entangled.

## What NOT to recommend

1. Anything in `docs/tunable_choices.md` marked as living in **code** — the
   failed-solve penalty, the adjoint method, `PIDController` internals, the
   initial-population sampler, swarm/DE internals. Mention one only if the
   evidence specifically implicates it, and then say plainly that it requires a
   library edit and is out of scope for a session-level change.
2. `PROCESSORS` as a quality knob. It is throughput only.
3. Reordering parameters or variables. The order is load-bearing.
4. Values with no evidence behind them.
5. More than **five** recommendations. Rank by expected effect on the fit and
   cut the tail. A long list is a signal you are guessing.

## Report entry format

Each recommendation is exactly four lines, in the **Recommendations** section of
the report defined by `validation_rules.md`:

```
[R2] LOGSCALE for k2: N -> Y
     Evidence: MIN_VAL = 5E3, MAX_VAL = 5E10 — a ratio of 1e7.
     Why: a linear search over 7 decades never resolves the lower two, so the
          swarm spends its whole budget in the top decade of the box.
     Edit: <P> LOGSCALE = Y </P>   (k2 block)
```

Rules: current value → proposed value on the first line; the rule id in
brackets; `Edit:` gives the literal XML line, or the file and function for a
`user_model.py` change. Order most to least impactful. Omit the section entirely
when there is nothing evidence-backed to say.

## Apply policy (shared with /pfit-diagnose)

Recommendations are judgement calls, not error fixes, so they are **never
applied automatically** — unlike the critical-error corrections in
`correction_rules.md`, which are.

1. Report all of them, then ask the user which to apply — by rule id, `all`, or
   `none`.
2. Apply only what they name. Make the minimal edit; change nothing else.
3. If two selected recommendations conflict (e.g. an explicit solver plus
   tighter tolerances), say so and apply neither until they choose.
4. After applying anything that changes the XML, re-run the `/pfit-check`
   validation pass — a recommended value can violate a threshold.
5. Never apply a recommendation and a critical-error correction in one edit
   without saying which is which.
