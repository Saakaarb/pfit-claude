# optax API digest (gradient stage)

**Pinned to: optax==0.2.8, jax==0.6.2**

Auto-generated from the installed packages by `tools/gen_api_context.py`. Do not edit by hand — edit `tools/gen_api_context.yaml` and regenerate.

If the versions above differ from the installed ones, this digest is stale: regenerate it before trusting it.

## Optimizers

| Name | Signature (defaults shown) |
|---|---|
| `optax.lbfgs` | `(learning_rate=None, memory_size=10, scale_init_precond=True, linesearch=<default>)` |
| `optax.adam` | `(learning_rate, b1=0.9, b2=0.999, eps=1e-08, eps_root=0.0, mu_dtype=None, nesterov=False)` |
| `optax.adamw` | `(learning_rate, b1=0.9, b2=0.999, eps=1e-08, eps_root=0.0, mu_dtype=None, weight_decay=0.0001, mask=None, nesterov=False)` |
| `optax.sgd` | `(learning_rate, momentum=None, nesterov=False, accumulator_dtype=None)` |
| `optax.rmsprop` | `(learning_rate, decay=0.9, eps=1e-08, initial_scale=0.0, eps_in_sqrt=True, centered=False, momentum=None, nesterov=False, bias_correction=False)` |
| `optax.adagrad` | `(learning_rate, initial_accumulator_value=0.1, eps=1e-07)` |
| `optax.nadam` | `(learning_rate, b1=0.9, b2=0.999, eps=1e-08, eps_root=0.0, mu_dtype=None, nesterov=True)` |
| `optax.adabelief` | `(learning_rate, b1=0.9, b2=0.999, eps=1e-16, eps_root=1e-16, nesterov=False)` |

## Learning-rate schedules

| Name | Signature (defaults shown) |
|---|---|
| `optax.exponential_decay` | `(init_value, transition_steps, decay_rate, transition_begin=0, staircase=False, end_value=None)` |
| `optax.cosine_decay_schedule` | `(init_value, decay_steps, alpha=0.0, exponent=1.0)` |
| `optax.linear_schedule` | `(init_value, end_value, transition_steps, transition_begin=0)` |
| `optax.warmup_exponential_decay_schedule` | `(init_value, peak_value, warmup_steps, transition_steps, decay_rate, transition_begin=0, staircase=False, end_value=None)` |
| `optax.piecewise_constant_schedule` | `(init_value, boundaries_and_scales=None)` |

## Utilities

| Name | Signature (defaults shown) |
|---|---|
| `optax.apply_updates` | `(params, updates)` |
| `optax.value_and_grad_from_state` | `(value_fn)` |

## Gotchas (curated — these are the ones that bite)

- `optax.lbfgs()` has a NON-STANDARD update signature. It requires the params, value, grad and value_fn: `optimizer.update(grad, state, params, value=value, grad=grad, value_fn=f)`. Calling `update(grad, state)` as with adam raises a TypeError. See `lib/algorithms/NODE/classes.py:354`.
- `optax.lbfgs()` performs its own line search, so it ignores the init_value_lr / end_value_lr / decay_rate_lr settings entirely. Those config fields only take effect when gradient_optimizer: adam.
- L-BFGS converges in tens of iterations; adam needs hundreds to ~1000. Setting num_iters for one optimizer and then switching to the other is a common misconfiguration.
