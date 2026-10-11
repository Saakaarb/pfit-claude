# Tunable fitting choices

This page is the human-facing index of settings an agent may recommend. The
complete schema, types, and defaults live in
`lib/LLM/reference/yaml_format.md`; the evidence required for a recommendation
lives in `lib/LLM/reference/tuning_rules.md`.

## Model and data choices

- Trainable parameter bounds and logarithmic scaling determine the search
  space. Changing them requires a new full fit.
- Fixed parameters, initial conditions, experiment declarations, column roles,
  forcing inputs, and observable mappings define the scientific problem.
- The loss function is implemented in `generated/user_model.py`. Changing the
  model or loss requires checking and JAX translation again.

These choices must reflect scientific information supplied by the user or an
identified source. They are not generic optimizer tuning knobs.

## ODE solver choices

- `integrator` selects the Diffrax solver.
- `dt0`, `rtol`, `atol`, and `max_steps` control initial step size, adaptive
  error tolerances, and the solve limit.
- Solver choice and tolerances trade runtime against integration accuracy and
  robustness. A change to them requires regeneration and a full fit.

Only solver names listed in `lib/LLM/api/diffrax.md` are valid for the pinned
environment.

## Population-stage choices

- `algorithm` selects differential evolution or particle swarm optimization.
- `population_size`, `num_iters`, and `processors` control breadth, budget, and
  local parallelism.
- The population optimizer's seed and algorithm-specific settings control
  reproducibility and exploration behavior.

Changes to the search bounds, population algorithm, or population budget
require a full fit. A larger population or iteration budget costs more function
evaluations and should be justified by observed search behavior.

## Gradient-stage choices

- `gradient_optimizer` selects the supported refinement algorithm.
- Learning rate or schedule, iteration budget, and convergence controls govern
  the local refinement.
- Gradient-only refinement is appropriate when a prior population run found a
  useful basin and only gradient-stage settings changed.

Only optimizer names and options listed in `lib/LLM/api/optax.md` are valid for
the pinned environment.

## Output and diagnostic choices

- `output.write_results` must remain enabled for auditable result tables and
  plots.
- Post-fit sloppiness analysis evaluates local parameter identifiability; it
  does not change the fitted result.
- Live-view settings affect monitoring, not the numerical objective.

Agents must quote measured evidence for recommendations, ask before applying a
scientifically meaningful change, and follow the apply policy in
`lib/LLM/reference/tuning_rules.md`.
