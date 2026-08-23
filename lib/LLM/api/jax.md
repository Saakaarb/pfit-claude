# jax / jax.numpy API digest (translation rules)

**Pinned to: jax==0.6.2, jaxlib==0.6.2, numpy==2.2.6**

Auto-generated from the installed packages by `tools/gen_api_context.py`. Do not edit by hand — edit `tools/gen_api_context.yaml` and regenerate.

If the versions above differ from the installed ones, this digest is stale: regenerate it before trusting it.

## Transforms

| Name | Signature |
|---|---|
| `jax.jit` | `(fun, in_shardings=UnspecifiedValue, out_shardings=UnspecifiedValue, static_argnums=None, static_argnames=None, donate_argnums=None, donate_argnames=None, keep_unused=False, device=None, backend=None, inline=False, abstracted_axes=None, compiler_options=None)` |
| `jax.grad` | `(fun, argnums=0, has_aux=False, holomorphic=False, allow_int=False, reduce_axes=())` |
| `jax.value_and_grad` | `(fun, argnums=0, has_aux=False, holomorphic=False, allow_int=False, reduce_axes=())` |
| `jax.vmap` | `(fun, in_axes=0, out_axes=0, axis_name=None, axis_size=None, spmd_axis_name=None)` |
| `jax.pmap` | `(fun, axis_name=None, in_axes=0, out_axes=0, static_broadcasted_argnums=(), devices=None, backend=None, axis_size=None, donate_argnums=(), global_arg_shapes=None)` |

## `jax.numpy` functions available for translation

The generated code translates numpy pseudocode to these. A numpy function absent from this list should be checked against the installed `jax.numpy` before use — several numpy APIs have no jnp equivalent.

**reductions**

| Name | Signature |
|---|---|
| `jnp.mean` | `(a, axis=None, dtype=None, out=None, keepdims=False, where=None)` |
| `jnp.sum` | `(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None, promote_integers=True)` |
| `jnp.max` | `(a, axis=None, out=None, keepdims=False, initial=None, where=None)` |
| `jnp.min` | `(a, axis=None, out=None, keepdims=False, initial=None, where=None)` |
| `jnp.sqrt` | `(x)` |
| `jnp.square` | `(x)` |
| `jnp.abs` | `(x)` |
| `jnp.log` | `(x)` |
| `jnp.log10` | `(x)` |
| `jnp.exp` | `(x)` |

**nan safe**

| Name | Signature |
|---|---|
| `jnp.isnan` | `(x)` |
| `jnp.nan_to_num` | `(x, copy=True, nan=0.0, posinf=None, neginf=None)` |
| `jnp.nanmax` | `(a, axis=None, out=None, keepdims=False, initial=None, where=None)` |
| `jnp.nanmean` | `(a, axis=None, dtype=None, out=None, keepdims=False, where=None)` |
| `jnp.nansum` | `(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None)` |
| `jnp.where` | `(condition, x=None, y=None, size=None, fill_value=None)` |

**array**

| Name | Signature |
|---|---|
| `jnp.array` | `(object, dtype=None, copy=True, order='K', ndmin=0, device=None)` |
| `jnp.concatenate` | `(arrays, axis=0, dtype=None)` |
| `jnp.stack` | `(arrays, axis=0, out=None, dtype=None)` |
| `jnp.clip` | `(arr=None, min=None, max=None, a=Deprecated, a_min=Deprecated, a_max=Deprecated)` |
| `jnp.zeros` | `(shape, dtype=None, device=None)` |
| `jnp.ones` | `(shape, dtype=None, device=None)` |
| `jnp.arange` | `(start, stop=None, step=None, dtype=None, device=None)` |
| `jnp.interp` | `(x, xp, fp, left=None, right=None, period=None)` |

**nonsmooth**

| Name | Signature |
|---|---|
| `jnp.sign` | `(x)` |
| `jnp.maximum` | `(*args, out=None, where=None)` |
| `jnp.minimum` | `(*args, out=None, where=None)` |
| `jnp.tanh` | `(x)` |

## Gotchas (curated — these are the ones that bite)

- `jax.config.update('jax_enable_x64', True)` must appear before any array is created. Without it every solve silently runs in float32 and stiff integrations lose accuracy.
- No Python control flow on traced values. `if failed: ...` fails to trace; use `jnp.where(failed, constants['error_loss'], loss_value)`.
- No in-place assignment. Use `arr.at[i].set(v)`, not `arr[i] = v`.
- NaN poisons reverse-mode autodiff even when multiplied by zero: the gradient of `jnp.where(mask, x, 0.0)` is NaN if `x` is NaN. Sanitise the data BEFORE the arithmetic: `safe = jnp.where(mask, dataset, 0.0)`, then divide. See `lib/LLM/reference/staggered_data.md`.
- Array shapes are static. Boolean-mask indexing (`x[mask]`) does not compile; multiply by a 0/1 mask and divide by `mask.sum()` instead.
