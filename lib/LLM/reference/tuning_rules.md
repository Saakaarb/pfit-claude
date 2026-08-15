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

Severity thresholds on the numbers already in the config belong to
`validation_rules.md`; this file owns recommendations **derived from the model
and the data**, i.e. from evidence the config does not contain. The human-facing
catalogue of what is tunable at all is `docs/tunable_choices.md`.

## The rule that governs every recommendation

**Every recommendation must cite the evidence that produced it.** A suggestion
with no evidence line is noise — omit it and let the default stand. Evidence
means a specific thing you read: a line of the RHS, a column range in the CSV,
a parameter's `min_val`/`max_val` ratio, a count. Never recommend a value
because it is "typical".

If the current config value is already reasonable, say nothing about that field.

**Every recommendation must also satisfy the cold-start invariant in
`cold_start.md`**: it must stand up from the equations, the dataset and the
bounds alone, with no reference solution. That file owns the invariant, what
counts as a solution artifact, and the substitutes for answer-derived evidence —
apply it from there. Every rule below is written to be computable under it.

## Evidence to gather first

Read these before writing any recommendation:

| Evidence | From | Used for |
|---|---|---|
| Whether the RHS is smooth | `user_defined_system` in `user_model.py` — look for `sign`, `abs`, `floor`, `ceil`, `clip`, `minimum`/`maximum`, `where` on a state-dependent condition | integrator family |
| How many discontinuities, and of what kind | the same expressions — count the distinct switches, and decide for each whether it is **one-way** (a threshold the trajectory crosses once and does not return through) or **chattering** (a switch the trajectory can re-cross repeatedly, as in a friction or on/off control law) | integrator family |
| Whether the fastest mode is **slaved or active** | the RHS **form** — structural, so it needs no parameter values, but it must be *derived, not recognised*. See the test below; do not classify a mode from the shape of its terms | integrator family |
| Worst-case stiffness **over the bounds** | evaluate the RHS's rate terms at the corners of the `min_val`/`max_val` box, not at any single parameter set — the global search visits the whole box, so the question is whether the system *can* be stiff anywhere in it. Quote the implied step count over the data's time span | integrator family, `max_steps` |
| Timescale evidence in the data | column 0 of each CSV against how fast each observable moves: an observable that is flat across most of the record and then changes by its full range within one or two sample intervals is direct evidence of a fast/slow split, measurable before any fit | integrator family, tolerances |
| Stiffness indicators | slaved fast modes, worst-case box stiffness and the data evidence (all above); whether the RHS mixes fast and slow variables; whether any bound in the config spans >4 decades | integrator family, tolerances |
| Time-span vs. sampling | first/last value and spacing of column 0 of each CSV; whether spacing is uniform or geometric | `max_steps`, `initial_timestep` |
| Per-column magnitudes | max `|value|` of each data column | loss normalisation and weighting |
| Blank/NaN cells | any empty cell in a CSV | staggered-data handling |
| Bound width per parameter | `max_val / min_val` per trainable parameter | `logscale` |
| N, number of trainable parameters | config | search budget, identifiability warning |
| Number of experiments | count of `experiments` blocks | aggregation caveat |

## Rules

### R1 — Integrator family (highest value; always emit a verdict)

This is the only recommendation to make even when the config already names a
solver, because a wrong family makes every other setting irrelevant.

Recommend a **family**, then name a specific solver only by reading it out of
the solver table in `lib/LLM/api/diffrax.md` — its `Kind` column gives the
family and its `Validated` column says which are exercised by this framework's
examples. Prefer a validated one. Never name a solver from memory, and never
recommend one absent from that table.

Smoothness and stiffness are **two axes, weighed by magnitude — not a gate and
a tie-breaker.** Decide both before naming a family; a rule that checks only one
of them will confidently give the wrong answer whenever the two conflict.

**Why they do not trade off symmetrically.** Every adaptive solver takes
`dt = min(stability limit, accuracy limit)`.

- An explicit method's stability limit is set by the *fastest eigenvalue in the
  system*, whether or not that mode is still doing anything. It is always
  present and always binding once that eigenvalue is large.
- An implicit method is stable across the whole decaying half-plane, so the
  stability term effectively drops out and the step is set by accuracy alone —
  by the timescale of whatever is actually moving.

The gap between those two *is* stiffness, which is why the payoff from an
implicit method lands **in** the stiff regions rather than outside them, and why
only a **slaved** fast mode counts. A fast mode that is still active is genuinely
fast dynamics: it must be resolved for accuracy by any method, and no solver
choice escapes it. A fast mode that has saturated costs an explicit method every
step while contributing nothing to the answer — that, and only that, is what an
implicit method converts into larger steps.

The two failure modes also differ in *extent*, which is what makes them
weighable at all:

- Non-smoothness is **local and transient**. The implicit inner solve fails only
  at the switching instants; the controller responds by collapsing the step
  there. The cost is bounded by how often the trajectory crosses a switch.
- Stiffness is **global and persistent**. It constrains every step over the whole
  interval, and there is no adaptive escape from it.

### Testing for a slaved mode

Slavedness decides whether an implicit method has anything to buy, so derive it
rather than recognising it. **Differentiate.**

The matrix to differentiate is `d f_i / d y_j`, the derivative of the RHS with
respect to the **state**. Because the ODE maps a state to a state-derivative
this is always **square**, `n_states x n_states`, and it is a symbolic
expression in the states and parameters — obtainable without any parameter
values. Do not confuse it with the sensitivity of the loss to the *parameters*,
which is a different, non-square object belonging to identifiability (R9) and
computed after a fit, not before one.

Start with the diagonal entry for the state `y_i` carrying the fastest term:

```
ratio = |f_i| / |d f_i / d y_i|
```

- If that ratio is the state's **remaining excursion** — the distance it still
  has to travel — then the derivative is large only while the state is still
  moving. The two vanish together, and **the mode is never slaved.** This is
  saturation, not stiffness: genuine fast dynamics that every method must
  resolve for accuracy, and an implicit solver buys nothing.
- If the ratio **collapses** while the derivative stays large — the state has
  arrived and is held there by a fast restoring term — the mode **is slaved**,
  and that is the stiffness an implicit method converts into larger steps.

Do **not** classify from the shape of the terms. A factor that drives its own
rate to zero (a reactant approaching exhaustion, `(1-x)**m` as `x -> 1`, a
relaxation towards an equilibrium) *looks* like the slaved case and usually is
not: for `m > 1` such a term's derivative vanishes faster than the term, so the
mode decelerates as it saturates. The exponent decides it, and only
differentiating reveals that.

Check the **exponent range the bounds allow**, not one value: a family like
`(1-x)**m` is non-stiff for `m` comfortably above 1 and genuinely slaved as
`m -> 1`, where the derivative tends to a constant while the term vanishes. If
the box admits the slaved corner at all, stiffness is *possible*, which is the
"stiffness unknown" case below — not a tie. Say which corner, and how much of
the box it is.

The diagonal ratio is a **proxy**, good only where a state chiefly limits its
own rate. It says nothing about the spectrum when the off-diagonal coupling is
comparable — watch especially for a row whose off-diagonals carry a large gain
from elsewhere in the model (an enthalpy, a stoichiometric factor, a
compartment-volume ratio), since such a factor scales the coupling without
scaling the diagonal. Report the sparsity pattern as well: states absent from
each other's equations give structural zeros that split an `n x n` into small
blocks.

Where the proxy is not clearly safe, bound the spectrum properly.

### Bounding the spectrum over the box

**The decision scalar.** Stiffness is not a property of the equations alone but
of (equations, states, horizon). Reduce it to one number:

```
N_explicit = t_span * max_i |Re lambda_i(J)|
```

the steps an explicit method needs for **stability**. Compare it against the
steps needed for **accuracy** (R10's `t_span / tau_min`). Stiffness is the gap.
A unitless "stiffness ratio" is not decision-relevant; this is.

**The tool: the matrix measure (Bendixson).** For real `J`,

```
lambda_min(J_sym)  <=  Re lambda_i(J)  <=  lambda_max(J_sym),
                                     J_sym = (J + J^T)/2
```

This converts a nonsymmetric eigenvalue problem — badly behaved under
perturbation — into a symmetric one that is Lipschitz-stable in the entries.
That stability is the whole reason to prefer it. One symmetric eigendecomposition,
`O(n^3)`.

**How to evaluate it without a solution.** Sample `(y, p)` over the state
enclosure and the parameter box, form `J` by automatic differentiation, and take
the extreme of `lambda_max(J_sym)` and `|lambda_min(J_sym)|` across the sample.

Sampling the **Jacobian** is not the trial integration forbidden by
`cold_start.md`. No solver runs, no trajectory is produced, and a singular or
inert parameter set is as informative as any other — `J` is defined pointwise
wherever the RHS is. The objection to sampling trajectories does not transfer.

**The asymmetry that makes this cheap.** The two claims have different burdens:

- *"stiff somewhere in the box"* needs **one witness** — sampling suffices, and
  finding one ends the analysis.
- *"not stiff anywhere in the box"* needs a **two-sided enclosure**. Build an
  interval matrix `[J]` over the box and apply **Rohn**:
  `lambda_max <= lambda_max(Jc) + rho(Delta)`, with `Jc` the centre and `Delta`
  the radius matrix. `O(n^3)`.

Only the tie-break branch below needs the certificate; every other branch is
settled by a witness.

**What not to bother with.** Vertex enumeration over the interval matrix
(Hertz) is exact but costs `2^(n-1)`, and it does not repay that: the
conservatism in this pipeline comes from the interval enclosure, not from the
eigenvalue step, so it returns the same answer as Rohn. If an enclosure is too
loose, tighten the *enclosure* — affine arithmetic or Taylor models, which track
the fact that one parameter appears in many entries — rather than paying for
exact eigenvalues of a loose box.

**Two caveats to state whenever you quote a bound.**

- **Non-normality.** `lambda_max(J_sym)` over-estimates the spectral abscissa,
  sometimes badly. So a *small* value is a certificate of non-stiffness, while a
  *large* value is only a warning. This is the honest direction: a strongly
  non-normal system with benign eigenvalues really can force small explicit
  steps through transient growth.
- **The state enclosure.** The bound is only as good as the box of states `y`
  you evaluated over. Say what you assumed; a generous box is conservative and
  acceptable, an arbitrary one is not.

**Before any of it, try the structural argument.** Non-dimensionalise, or look
for singularly-perturbed form `eps * dy/dt = g(...)`: the interval on `eps`
implied by the bounds is the stiffness estimate, with no eigenvalue computation,
and it names *which parameter is responsible* — which the numerical bounds do
not. Do not, however, use a bare ratio of rate constants as a bound: rate
constants that reach the Jacobian through a product with another fitted
parameter make it an under-estimate, which is the unsafe direction.

Resolve on both axes, in this order:

- **Slaved fast modes present (stiff) and the discontinuities are few and
  one-way** → recommend an **implicit** solver despite the non-smoothness. The
  Newton failures are confined to a handful of crossings and are survivable,
  whereas the stiffness would pin an explicit method's step for the entire
  solve. State the step counts that make the comparison concrete — derived from
  the box corners and the data's time span, since no fitted parameters exist
  yet.
- **Non-smooth with chattering switches and no slaved fast mode** → recommend an
  **explicit** solver. Cite the exact offending expression. Explain that an
  implicit inner solve cannot converge across the discontinuity for *any*
  `max_steps`, so failures there are not fixable by raising it.
- **Stiff *and* chattering** → this is a genuine conflict and neither family is
  right. Say so plainly rather than picking the lesser evil, and recommend
  fixing it in the **model**: replace the hard switch with a smooth transition
  over a physically meaningful width, which restores the implicit solver the
  stiffness demands. Note that this is a change to `user_model.py`, so it
  requires re-running the check and translation steps.
- **Smooth RHS with a wide timescale ratio, or stiffness unknown** → recommend an
  **implicit** solver. Note the asymmetry: implicit on a non-stiff smooth system
  costs ~3–10× per step but still converges.
- **Both families safe** → recommend the **explicit** one. This is the
  tie-break: once neither family risks failing, the only remaining difference is
  cost, and an implicit method pays for a nonlinear solve at every step that it
  is no longer buying anything with. "Safe" means demonstrated, not assumed —
  the RHS is smooth (or its switches are rare and one-way) *and* stiffness has
  been **bounded**, not merely left unexamined. Stiffness that is simply unknown
  is not a tie; that case is covered above and goes to implicit.

Having settled the family, say a word about **order and structure**, which the
family alone does not determine:

- Severe stiffness rewards a **high-order** implicit method: the step is limited
  by accuracy, so a higher order buys a proportionally larger step for the same
  error, and the per-step Newton cost is amortised over far fewer steps. Read the
  available orders out of the digest rather than assuming a family has one.
- When the RHS splits cleanly into a stiff part and a non-stiff part, the digest
  may list an **IMEX** (implicit-explicit) method that treats each part
  appropriately. Recommend one only if the split is visible in the equations and
  the digest actually offers such a solver.

### R2 — `logscale` per parameter

`max_val / min_val` >= 100 → recommend `logscale: true`, quoting the ratio.
Below that, linear is fine. State it per parameter by name; do not issue a
blanket recommendation for all of them.

### R3 — Bounds

- A bound that is not strictly positive on a parameter marked `logscale: true` is
  a critical error, not a recommendation — hand it to `validation_rules.md`.
- Bounds spanning more than ~8 decades → recommend narrowing using whatever the
  source or the data implies, and cite what that is. A 10-decade box wastes most
  of the population budget.
- If the source document (for a `/pfit-new` session) gives a literature
  value, recommend a box bracketing it by 1–2 decades rather than an arbitrary
  one.

### R4 — Two-tier tolerances

If `population_opt.stepsize_rtol`/`_ATOL` are absent, recommend setting them ~100× looser
than `stepsize_rtol`/`_ATOL`, and say what that buys: the global search does
`population_size × num_iters` solves and does not need refinement accuracy.
Include the concrete values for this config, one per state variable.

Do not recommend this when the loss is dominated by a fast transient that loose
tolerances would smear — if the CSV's early time spacing is orders of magnitude
finer than its late spacing, say so and recommend only ~10× looser.

### R5 — Search budget

Recommend `population_size` >= 10·N and `num_iters` >= 20, N = number of
trainable parameters. Report the implied number of ODE solves
(`population_size × num_iters`), and — if a per-solve time is known from a
previous session — the implied wall-clock at the current `processors`. Budget is
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
  must also recommend an iteration count and an LR schedule (start ~1e-3, end
  ~1e-5) in the same entry — `adam` with L-BFGS-sized iteration counts does
  nothing.

  Set the count from the learning rate, not from a fixed number:
  **`num_iters` >= `1 / init_value_lr`**, which is 1000 at the recommended
  starting rate. The reason is that adam's step is normalized — it moves about
  `lr` per iteration in the `[-1, 1]` scaled parameter space whatever the
  gradient magnitude — so `num_iters * init_value_lr` is the distance it can
  cover, and a count that cannot cover a basin cannot reach its bottom.
  `validation_rules.md` owns the threshold and the warning.
- Never recommend the LR fields while `gradient_optimizer` is `lbfgs`; they are
  silently ignored.

### R8 — Loss shape

The loss lives in `user_model.py`, so these are recommendations about code, not
config. Quote the line you would change.

- Column magnitudes differing by more than ~10× and no per-column scaling in
  `_compute_loss_problem` → recommend the per-column-scaled RMSE pattern from
  `user_model_contract.md`, citing the two column ranges. Without it the largest
  column silently owns the fit.
- Any blank cell in a CSV and no NaN handling in the loss → recommend the
  NaN-safe pattern from `staggered_data.md`. Note that sanitising *after* any
  arithmetic still produces NaN gradients.
- More than one `experiments` and datasets of visibly different quality or
  length → note that aggregation is an unweighted mean over experiments with no
  weighting available, so the user should decide whether that is acceptable
  (per-experiment weighting would require a code change).

### R9 — Identifiability, in advance

N > number of observed columns × 3, or two parameters that appear only as a
product/ratio in the RHS → note that these will likely show up as
non-identifiable directions in the post-fit sloppiness report, and that the fix
is fewer trainable parameters or a reparameterisation, not more iterations.
Cite the RHS expression where the parameters are entangled.

### R10 — `max_steps`

Size it from the data and the equations. There is no empirical confirmation step
available, because the failure code that would carry the signal is ambiguous —
see the end of this rule.

**The estimate.** Take `tau_min`, the fastest local timescale visible in the
data — for each observable, the smallest value of `|y| / |dy/dt|` using finite
differences on the column. Then

```
max_steps ~ t_span / tau_min
```

This assumes the solver runs at its smallest step for the entire span, so it
overshoots — usually by around an order of magnitude, since an adaptive
controller spends small steps only where the solution demands them. Overshooting
is the right direction for a ceiling. Do **not** instead sum `dt_i / tau_i` over
the samples: that is the adaptivity-aware count, and at cold start it
underestimates badly, because the sampling grid is almost always coarser than
the dynamics it is sampling.

Cross-check `tau_min` against the equations as well as the data, and take the
smaller: if a term in the RHS can produce motion faster than anything the
sampling grid could have recorded, the data's `tau_min` is an artifact of the
sampling rate rather than a property of the system. Evaluate that term's
timescale expression over the bounds — this is arithmetic on the RHS, not a
trial integration (`cold_start.md`).

**Do not tune it by sweeping.** Raising `max_steps` over sampled parameter sets
and watching the success count does not measure what it appears to: most of a
box is singular or inert, so the count is dominated by candidates that no budget
would rescue and by ones that never needed a large budget. Note also that
`max_steps_reached` does not distinguish *"needs a larger budget"* from *"is
singular and the step is collapsing toward zero"* — both burn the whole budget
and return the same code — so the count cannot be read as evidence either way.

State the estimate and the two numbers behind it. Because the cost of the global
search is roughly linear in `max_steps`, an over-large ceiling is paid on every
doomed candidate; prefer the estimate above to an arbitrary large round number.

1. Anything in `docs/tunable_choices.md` marked as living in **code** — the
   failed-solve penalty, the adjoint method, `PIDController` internals, the
   initial-population sampler, swarm/DE internals. Mention one only if the
   evidence specifically implicates it, and then say plainly that it requires a
   library edit and is out of scope for a session-level change.
2. `processors` as a quality knob. It is throughput only.
3. Reordering parameters or variables. The order is load-bearing.
4. Values with no evidence behind them.
5. More than **five** recommendations. Rank by expected effect on the fit and
   cut the tail. A long list is a signal you are guessing.
6. Anything that presumes the answer — a setting justified by a reference
   trajectory, a published parameter table, or a value the source arrived at by
   fitting. See `cold_start.md`. Such a value is legitimate as a *starting box*
   (R3) but never as evidence that a setting is correct.

## Report entry format

Each recommendation is exactly four lines, in the **Recommendations** section of
the report defined by `validation_rules.md`:

```
[R2] logscale for k2: false -> true
     Evidence: min_val = 5.0e+03, max_val = 5.0e+10 — a ratio of 1e7.
     Why: a linear search over 7 decades never resolves the lower two, so the
          swarm spends its whole budget in the top decade of the box.
     Edit: model.trainable_parameters, k2 entry -> logscale: true
```

Rules: current value → proposed value on the first line; the rule id in
brackets; `Edit:` gives the literal config line, or the file and function for a
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
4. After applying anything that changes the config, re-run the `/pfit-check`
   validation pass — a recommended value can violate a threshold.
5. Never apply a recommendation and a critical-error correction in one edit
   without saying which is which.
