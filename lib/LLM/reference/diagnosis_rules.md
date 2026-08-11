---
topic: How to diagnose a completed fit from its outputs and recommend what to change
consumed_by: [pfit-diagnose]
generated: false
owns: >
  The post-fit evidence sources, the symptom-to-cause-to-fix rules, and the
  diagnosis report format.
---

# Diagnosis rules (post fit)

Applied by `/pfit-diagnose` to a session that has already been fitted. Pre-fit
recommendations are a different job — see `tuning_rules.md`, whose evidence rule,
"what NOT to recommend" list, report-entry format and apply-on-confirmation
policy all apply here unchanged. This file owns only the post-fit evidence
sources and the symptom rules.

## Evidence sources

Every one of these is in `sessions/<session>/outputs/`. Read all that exist
before drawing any conclusion — several symptoms are only distinguishable by
combining two artifacts.

| Artifact | Format | What it tells you |
|---|---|---|
| `pso_fitting.log` / `de_fitting.log` | header, then `iter, best_cost, seconds` per line | stage-1 trajectory: did the cost move, when did it flatten, seconds per iteration |
| `NODE_fitting.log` | header, then `iter, best_loss, seconds` per line | stage-2 trajectory. Note it logs *best-so-far*, so it is monotone by construction — flatness means no improvement, not divergence |
| `final_design_point.csv` | one real-unit value per line, XML trainable order | the answer; compare each value against its `MIN_VAL`/`MAX_VAL` |
| `sloppiness_report.txt` | see `project_context.md` | loss at best fit, `|grad|_inf`, eigenvalue spectrum, spread in decades, count of non-identifiable directions, stiffest/sloppiest eigenvectors |
| `result_solution_expN.csv` | `time | data columns | solution columns` (written only when `WRITE_RESULTS = Y`) | per-column, per-time residuals — the only way to see *which* observable and *which* time region is being missed |
| `fit_result*.png` | plot | quick confirmation of what the residuals say |

Cross-check against `inputs/user_input.xml` for the settings that produced them,
and note the failed-solve penalty is `error_loss = 5000.0`
(`lib/utils/xmlread.py`).

## Symptom rules

Work top to bottom; the earlier symptoms invalidate the later ones — S1–S3 all
make the numbers below them untrustworthy, so a hit there ends the analysis of
the stage-1 log rather than adding to it. If S1 or S2
fires, stop — the numbers downstream of a failing solve mean nothing.

### S1 — Every solve is failing

*Evidence:* stage-1 `best_cost` is `5.0000E+03` (or within float noise of it) on
every line, or `NODE_fitting.log` shows the run exiting after one iteration.

*Cause, in order of likelihood:* wrong integrator family for the RHS;
`MAX_STEPS` too small; bounds admitting parameter values that make the system
unintegrable; an `INITIAL_TIMESTEP` far too large for the initial transient.

*Fix:* re-apply `tuning_rules.md` R1 against the RHS. Then raise `MAX_STEPS`
(×10) — **but only if the RHS is smooth**; for a non-smooth RHS raising it
cannot help, and the integrator must change instead. Report which of the two
cases this session is in; do not offer both as equal options.

### S2 — Some solves are failing

*Evidence:* stage-1 `best_cost` sits at `5.0000E+03` for the first iterations
then drops; or the final loss is good while `sloppiness_report.txt` shows an
implausibly large `|grad|_inf`.

*Cause:* part of the search box is unintegrable, so the optimizer is exploring
into a wall.

*Fix:* narrow the offending bound, or raise `MAX_STEPS`. Say that the invisible
region is a silent constraint on the search — this is worth fixing even though
the fit "worked".

### S3 — The two stages disagree about the same point

*Evidence:* stage 1's final `best_cost` versus the **first** loss logged in
`NODE_fitting.log`. Both describe nearly the same parameter vector, since stage 2
is seeded from stage 1's best point. Corroborate with the decade gap between
`POP_STEPSIZE_*` and `STEPSIZE_*` in the XML — the ratio alone is not enough,
because `NODE_fitting.log`'s first line is already one optimizer step in and
L-BFGS's line search can legitimately take a large first step.

Two bands, and they get different treatment:

- **more than ~10×, or any gap of 3+ decades in tolerance:** a confident
  finding. Report it.
- **~2× to ~10× with a smaller tolerance gap:** report it as an observation only
  if the fit has another problem worth re-running for. A converged fit with good
  residuals and a 3× handoff gap is working as designed — the loose stage found
  the right basin, which is all it was asked to do. Do not recommend a re-fit for
  this alone.

Read both numbers before drawing any conclusion from the stage-1 trajectory,
whichever band you land in.

*Cause:* the two stages evaluate the loss at different integration accuracies.
`POP_STEPSIZE_RTOL`/`_ATOL` are looser than `STEPSIZE_RTOL`/`_ATOL` (or absent
and therefore inherited, in which case look for a stage-1 loss inflated by
partial solve failures instead). At the loose tolerance the global search was
not ranking candidates on the real loss but on a surrogate distorted by
integration error — so its "best" point is only best under that distortion, and
its whole trajectory is untrustworthy.

*Fix (upper band only):* tighten `POP_STEPSIZE_RTOL`/`_ATOL` — a factor of ~100
toward the refinement tolerances is the usual step — and re-run the **full** fit,
not the gradient stage alone: the seed point itself was chosen wrongly. Quote
both losses and the tolerance gap as the evidence. Say explicitly that this
invalidates the stage-1 diagnosis: do not also report S4 or S5 from the same log,
because the costs those rules read are the distorted ones.

The two-tier tolerance split is still the right structure (see `tuning_rules.md`
R4) — the recommendation is to narrow the gap, never to abolish it by deleting
the `POP_STEPSIZE_*` fields.

*Reverse direction:* stage 1's final `best_cost` much **lower** than the first
NODE loss is the same disagreement with the same fix. It is the more dangerous
form, because the reported stage-1 result looks better than the fit really is.

### S4 — Stage 1 flattened early

*Evidence:* `best_cost` in the stage-1 log stops improving before ~60% of
`NUM_ITERS`, at a value well above the final NODE loss.

*Cause:* premature convergence onto one basin; population too small for N.

*Fix:* raise `POPULATION_SIZE` toward 10·N or beyond before raising `NUM_ITERS`
— extra iterations of a collapsed swarm buy nothing. If it recurs with an
adequate population, recommend `ALGORITHM = DE` and note that changing
`RANDOM_SEED` samples a different basin, which is a diagnostic, not a fix.

### S5 — Stage 1 never flattened

*Evidence:* `best_cost` was still improving materially on the last logged
iteration.

*Fix:* `NUM_ITERS` was the binding constraint — raise it. Quote the last few
cost values as the evidence that budget, not algorithm, was the limit.

### S6 — Gradient stage did nothing

*Evidence:* `NODE_fitting.log`'s final `best_loss` is within a few percent of
stage 1's final `best_cost`, over more than a handful of iterations.

*Cause, distinguish these — they need opposite fixes:*

- **Loss not normalised.** Check the loss magnitude in
  `sloppiness_report.txt`; a loss far from O(1) means the step sizes are
  meaningless. Fix in `_compute_loss_problem` per `user_model_contract.md`.
- **Gradients are inaccurate.** `STEPSIZE_RTOL`/`_ATOL` too loose for
  differentiation. Tighten them; note the forward solve can be accurate enough
  to rank candidates while being too loose to differentiate.
- **Parameters pinned at a bound.** Compare `final_design_point.csv` to the
  bounds. The refinement stage hard-clips to the box, so a pinned parameter
  stops moving while still carrying a gradient — it looks converged and is not.
  Widen that bound.
- **Too few iterations for the optimizer chosen.** `adam` with `NUM_ITERS` in
  the tens cannot converge. Either raise it to >=200 or switch to `lbfgs`.

State which one the evidence supports. If the evidence cannot separate them, say
so and give the cheapest discriminating experiment.

### S7 — Fit is good on one observable, bad on another

*Evidence:* per-column residuals from `result_solution_expN.csv`. Compute them;
do not judge from the plot alone.

*Cause:* unweighted or unscaled loss, so the large-magnitude column dominates;
or a wrong observable → state mapping if one column is badly wrong at *every*
time.

*Fix:* per-column scaling or explicit weighting in `_compute_loss_problem`.
A column that is wrong everywhere, including t=0, points at the mapping instead
— flag that as the more likely cause, since no amount of weighting fixes it.

### S8 — Fit is good early and bad late (or the reverse)

*Evidence:* residuals grouped by time region from `result_solution_expN.csv`.

*Cause:* a fast transient dominating a normalised RMSE and starving the slow
tail; or a missing/incorrect slow term in the model.

*Fix:* time-weighting in the loss, or fitting a log-spaced subset. Also say
plainly that a systematic late-time bias is often model structure, not settings
— more tuning will not fix a missing term.

### S9 — Sloppy or non-identifiable

*Evidence:* `sloppiness_report.txt` — spread > 6 decades, or a nonzero count of
non-identifiable directions.

*Fix:* read the SLOPPIEST eigenvector and name the parameter combination it
implicates. Recommend fixing the least-constrained parameter to a literature
value, or reparameterising to the combination the data actually constrains.
Explicitly say that raising either stage's iteration count will not help: the
direction is flat, so there is nothing to descend.

### S10 — It converged and it is slow

*Evidence:* seconds-per-iteration from either log, against the number of solves.

*Fix:* loosen `POP_STEPSIZE_*` (the global search does the overwhelming majority
of the solves); raise `PROCESSORS` toward the core count; consider an explicit
solver **only** if the RHS is smooth and the timescale evidence says non-stiff.
Never trade correctness for speed without saying that is the trade.

### S11 — Nothing is wrong

If the loss is small, the residuals are unstructured, no parameter is pinned, and
the sloppiness verdict is well-determined, say so in one line and stop. Do not
manufacture recommendations for a converged fit.

## Report format

Write to `sessions/<session>/outputs/fit_diagnosis.txt` and summarise in chat.

```
Fit diagnosis: <session>
================================================================
Stage 1 : <algorithm>, <N_iters> iters, best cost <x> -> <y>
Stage 2 : <optimizer>, <N_iters> iters, best loss <x> -> <y>
Sloppiness: <verdict>, spread <d> decades, <k> non-identifiable
Verdict : <one line: converged / limited by X / failing>

Findings
--------
[S6] Gradient stage improved the loss by 1.2% over 10 iterations.
     Evidence: NODE_fitting.log 2.68E-03 -> 1.13E-03 flat after iter 3;
               k2 = 3.01E+07 sits at MAX_VAL = 5.00E+07 (within 40%).
     Cause: parameter pinned at its upper bound; refinement hard-clips to the box.
     Edit: <P> MAX_VAL = 5E9 </P>   (k2 block)
```

Same rules as `tuning_rules.md`: every finding cites evidence, at most five,
ordered by impact, and nothing is applied until the user picks. Omit the
Findings section entirely on an S11 verdict.
