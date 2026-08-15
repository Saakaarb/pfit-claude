---
topic: The cold-start invariant — the agent never has the solution when it makes setup choices
consumed_by: [pfit-skeleton, pfit-from-source, pfit-check, pfit-jax, pfit-diagnose]
generated: false
owns: >
  The definition of a solution artifact, the setup/diagnosis phase split and
  what each may read, the three violation patterns, and the substitutes that
  replace answer-derived evidence (worst case over the bounds, the dataset as
  an instrument, the reachability check).
---

# The cold-start invariant

**Every setup choice is made without the solution. There are no exceptions, and
the assumption is never relaxed because a solution happens to be at hand.**

A user fitting a real problem has exactly three things:

1. the **model equations**, from a paper, textbook or their own derivation;
2. the **dataset**;
3. the **parameter bounds** — a prior, often a wide guess.

They do not have fitted parameters, a reference trajectory, or a target loss. If
they did, there would be nothing to fit. Every rule in this instruction layer
must therefore be applicable with those three inputs alone.

This holds even when the session in front of you *does* have a stored answer —
a re-run of a solved problem, a worked example, a source document that prints
its own fitted table. Availability is not permission. A rule validated against a
known answer is not known to work; it is only known to agree, and it will fail
silently the first time it meets a problem nobody has solved.

## What counts as a solution artifact

Off-limits while any setup choice is being made:

- `outputs/final_design_point.csv` — the fitted parameter vector
- `outputs/result_solution_exp*.csv` — the fitted trajectories
- `outputs/*_fitting.log`, `outputs/sloppiness_report.txt`, `outputs/fit_diagnosis.txt`
- the same artifacts belonging to **any other session**
- a fitted parameter table printed in a source document, and any value the
  source arrived at by fitting or by hand-pinning an unidentifiable parameter

The last one is the easiest to miss, because it arrives looking like part of the
problem statement. A number a source obtained *by solving* is a result. It may
seed a search box (see R3 in `tuning_rules.md`); it may never serve as evidence
that a setting is correct.

## The two phases

The invariant is a phase rule, not a blanket ban on reading `outputs/`:

| Phase | Skills | May read |
|---|---|---|
| **Setup** | `/pfit-skeleton`, `/pfit-from-source`, `/pfit-check`, `/pfit-jax` | equations, dataset, bounds. **No solution artifact of any session.** |
| **Diagnosis** | `/pfit-diagnose` | additionally, **this session's own** `outputs/` |

Diagnosis is not an exception to the invariant. What it reads is the record of
the fit that just ran — evidence about the optimizer's behaviour, not a
reference answer — and it still may not consult another session's results or a
published parameter table to decide whether this fit is right.

When a diagnosis leads back to a setup change, the setup rules apply again to
that change: what carries over is the *observed behaviour* of the run (a stalled
loss, a pinned parameter, an exit gradient), never a target to aim at.

## The three violation patterns

1. **Reasoning from a particular parameter set.** Deriving rates, timescales or
   stiffness from specific values. At cold start no such values exist.
2. **Treating a result as a given.** Using a source's fitted or hand-pinned
   value as though it were a property of the problem. Where such a value is
   genuinely needed to define the model, label it an assumption in the report
   and say what it is worth if wrong.
3. **Judging a setting by whether it reproduces a known answer.** The answer is
   the unknown; agreement with it measures nothing about the rule.

## Substitutes

Each replaces an answer-derived judgement with one computable from the three
available inputs.

### Worst case over the bounds

Where a quantity cannot be known before fitting, evaluate it at the corners of
the `min_val`/`max_val` box rather than at any single parameter set, and say
that is what you did.

This is not a weaker test but a stricter one, and it is the question that
actually matters: the global search visits the whole box, so a candidate that
fails anywhere in it scores `error_loss` — indistinguishable from a genuinely
bad parameter set. The optimizer cannot report the difference; it simply
converges confidently in whatever region still integrates.

### The dataset as an instrument

The data is measured, not inferred, so everything in it is available at cold
start and none of it presumes an answer: the sampling grid and span, each
column's range, how fast each observable moves between samples, whether a
channel is flat for most of the record and then moves within one interval. These
constrain the timescales the model must reproduce and the accuracy the solve
must deliver.

### Read the equations — do not sample the box

**The integrator family, `max_steps` and the tolerances are decided by reading
the equation terms and the dataset. Never by sampling parameters and
integrating.**

Trial integration at guessed parameters looks like empiricism and is not. On
exactly the hard problems where the decision matters, the overwhelming majority
of a box is non-physical: some parameter sets are singular and blow up in finite
time, and most of the rest are inert, so the model does nothing at all. A sample
is therefore dominated by trajectories that are either unsolvable or trivial,
and neither kind exercises the behaviour the setting has to serve. Timing or
success-rate comparisons over such a sample measure per-step overhead on
trivial solves, not the property being decided.

It is also circular: a sample must be integrated by *some* solver, so it cannot
adjudicate the choice of solver without assuming it.

What to read instead:

- **The equation terms.** Each term carries a characteristic timescale as an
  expression in the parameters — a rate constant, a relaxation coefficient, an
  exponential sensitivity. Evaluate those expressions symbolically over the
  bounds. This is arithmetic on the RHS, not integration, and it needs no
  trajectory.
- **The structure.** Whether a fast term is attached to a state that saturates
  (a slaved mode) or to one that is still moving is visible in the algebra
  alone, and it is what decides whether an implicit method can convert stiffness
  into larger steps.
- **The dataset.** The span, the sampling grid, each column's range, and how
  fast each observable moves between samples. This is measured, so it bounds the
  timescales the model must reproduce and the accuracy the solve must deliver —
  with no guessing.

Where the equations and the data disagree with each other, say so: an RHS that
cannot produce motion on the timescale the data shows is a modelling problem,
and it is visible from those two sources without fitting anything.
