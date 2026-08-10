---
topic: Fitting when observables are sampled at different time points
consumed_by: [pfit-skeleton, pfit-from-source, pfit-check, pfit-jax]
generated: false
owns: >
  The union-grid + NaN-masking layout, the t=0 anchor row, the interpolation
  alternative, and the validator exemptions that follow from them.
---

# Handling Staggered / Ragged Time-Series Data

## What this covers

This document describes how to set up a fitting problem when the observables are
NOT all sampled at the same time points. This is common in real biological data:
e.g. observable A is measured at t = {4, 6, 8} while observable B is measured at
t = {6, 8, 13, 28}. The framework's data format assumes a single rectangular
grid (column 0 = time, every other column = one observable, all sharing one time
vector), so ragged sampling must be reconciled to that format.

If every observable IS measured at the same times, ignore this document — just
build the rectangular CSV directly.

## Why you CANNOT simply split each observable into its own file

It is tempting to put each observable in its own CSV and use one EXPERIMENT
block per observable. This does NOT work. The framework's multiple-EXPERIMENT
mechanism is for the SAME observables measured under DIFFERENT conditions or
initial conditions; it calls `_compute_loss_problem(constants, trainable_variables)`
once per experiment and passes NO signal of which experiment/observable it is
(only `dataset`, `t_eval`, `init_cond`). With one observable per file, the loss
function cannot tell whether the single data column should be compared to state
variable 0, 1, 2, ... — they map to different species. So per-observable
splitting silently compares against the wrong state. Only split by EXPERIMENT
when the experiments share the same observable set (different conditions/ICs).

A single biological trajectory observed raggedly is therefore ONE experiment
with one data file, handled by one of the two approaches below.

## Approach 1 (RECOMMENDED): union grid + NaN masking

1. Build ONE data.csv whose time column is the UNION of all observables' time
   points (sorted). For each row/observable cell where that observable was NOT
   measured at that time, leave the cell BLANK (empty between commas). The
   framework loads data with `np.genfromtxt(..., delimiter=',')`, which reads a
   blank cell as NaN.

2. In `_compute_loss_problem`, mask the NaNs and normalise per column. Use this
   NaN-SAFE pattern (sanitise BEFORE any arithmetic, so NaN never enters the
   autodiff graph — otherwise gradients become NaN even where masked):

       mask = ~np.isnan(dataset)                       # True where measured
       scale = np.nanmax(np.abs(dataset), axis=0)      # per-observable peak
       data_safe = np.where(mask, dataset, 0.0)        # replace NaN with 0 FIRST
       resid = np.where(mask, (model_obs - data_safe) / scale, 0.0)
       loss = np.sqrt(np.sum(resid * resid) / np.sum(mask))

   In JAX (generated_script.py) this becomes the identical structure with
   jnp.isnan / jnp.nanmax / jnp.where / jnp.sum. The key rule: divide
   `(model - data_safe)`, never `(model - dataset)`, because `dataset` still
   contains NaN.

   Per-observable weighting note: `sum(resid^2)/sum(mask)` pools all points, so
   an observable with many samples dominates one with few. If equal weight per
   observable is wanted, average per-column RMSEs instead.

## Initial conditions earlier than the first measurement

Very common: the ICs are defined at t=0 but the first sample is at t=4.

Set `INITIAL_TIME = 0.0` in GRADIENT_OPT. Integration then starts at the ICs and
the solution is still saved at exactly `t_eval` (the `SaveAt` convention in
`jax_translation.md`), so the saved rows stay aligned with the data rows. No
anchor row is needed for alignment.

An anchor row at the IC time is still worth adding when it carries information —
a measured baseline at t=0 is real data and contributes a genuine residual. If
you add one purely to satisfy the layout, leave its observable cells BLANK (NaN)
so it contributes nothing to the loss.

> Historical note: an earlier convention saved at `[init_time, t_eval[1:]]`,
> which made an anchor row mandatory — without it, row 0 compared the model at
> `init_time` against the data at `t_eval[0]`. That convention has been removed;
> see the SaveAt section of `jax_translation.md`.

## Approach 2 (SIMPLER, but use with care): interpolate missing points

Linearly interpolate each observable onto the union grid so every cell is filled
and the CSV is fully rectangular with no NaN; the standard (unmasked) loss then
works unchanged. This avoids all masking code.

Trade-off: interpolation FABRICATES measurements. The optimiser then fits the
model to synthetic points lying on straight lines between true samples, which
biases the fit toward interpolation artifacts and understates noise. It is only
acceptable when the data are DENSE and SMOOTH (small interpolation error).
For SPARSE or NOISY series (few points, large swings between them), prefer
Approach 1 — fit only the real measurements. Never extrapolate beyond the first
or last real sample of an observable.

## Checklist for the validator (do NOT flag these as errors)

- Blank cells / NaN in data.csv are intentional under Approach 1; do not flag.
- A t=0 (or IC-time) anchor row with blank observables is intentional.
- `np.isnan` / `np.nanmax` in `_compute_loss_problem` are expected and are
  JAX-translatable (jnp.isnan / jnp.nanmax).
