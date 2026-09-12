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

| Name | Signature | Inputs / dtype contract |
|---|---|---|
| `jnp.mean` | `(a, axis=None, dtype=None, out=None, keepdims=False, where=None)` | `a`: ArrayLike. `where`: ArrayLike boolean mask or None. `dtype`: DTypeLike or None; output dtype matches floating inputs, otherwise promotes non-floating inputs to float32/float64. |
| `jnp.sum` | `(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None, promote_integers=True)` | `a`: ArrayLike. `initial`/`where`: ArrayLike or None; `where` is a broadcast-compatible mask. `dtype`: DTypeLike or None; integer inputs promote to widest available integer unless `promote_integers=False` or `dtype` is set. |
| `jnp.max` | `(a, axis=None, out=None, keepdims=False, initial=None, where=None)` | `a`: ArrayLike. `initial`/`where`: ArrayLike or None; `where` is a broadcast-compatible boolean mask and requires `initial`. |
| `jnp.min` | `(a, axis=None, out=None, keepdims=False, initial=None, where=None)` | `a`: ArrayLike. `initial`/`where`: ArrayLike or None; `where` is a broadcast-compatible boolean mask and requires `initial`. |
| `jnp.sqrt` | `(x)` | `x`: ArrayLike real or complex. Negative real inputs return NaN; complex inputs return complex. |
| `jnp.square` | `(x)` | `x`: ArrayLike numeric. Output follows JAX type-promotion rules. |
| `jnp.abs` | `(x)` | `x`: ArrayLike numeric. Complex input returns real magnitude. |
| `jnp.log` | `(x)` | `x`: ArrayLike numeric. Integers/bools are promoted to inexact dtype; negative real inputs return NaN. |
| `jnp.log10` | `(x)` | `x`: ArrayLike numeric. Integers/bools are promoted to inexact dtype; negative real inputs return NaN. |
| `jnp.exp` | `(x)` | `x`: ArrayLike numeric. Integers/bools are promoted to inexact dtype. |

**nan safe**

| Name | Signature | Inputs / dtype contract |
|---|---|---|
| `jnp.isnan` | `(x)` | `x`: ArrayLike numeric. Returns bool Array. |
| `jnp.nan_to_num` | `(x, copy=True, nan=0.0, posinf=None, neginf=None)` | `x`: ArrayLike. `nan`, `posinf`, `neginf`: ArrayLike replacements or None. Inexact inputs are sanitized; non-inexact inputs are returned unmodified. |
| `jnp.nanmax` | `(a, axis=None, out=None, keepdims=False, initial=None, where=None)` | `a`: ArrayLike. `initial`/`where`: ArrayLike or None; `where` is a broadcast-compatible boolean mask. |
| `jnp.nanmean` | `(a, axis=None, dtype=None, out=None, keepdims=False, where=None)` | `a`: ArrayLike. `where`: ArrayLike boolean mask or None. `dtype`: DTypeLike or None. |
| `jnp.nansum` | `(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None)` | `a`: ArrayLike. `initial`/`where`: ArrayLike or None; `where` is a broadcast-compatible boolean mask. `dtype`: DTypeLike or None. |
| `jnp.where` | `(condition, x=None, y=None, size=None, fill_value=None)` | `condition`: broadcast-compatible boolean ArrayLike. Three-argument form needs `x` and `y` ArrayLike, broadcast-compatible, typecast-compatible; result dtype is `jnp.result_type(x, y)`. |

**array**

| Name | Signature | Inputs / dtype contract |
|---|---|---|
| `jnp.array` | `(object, dtype=None, copy=True, order='K', ndmin=0, device=None)` | `object`: any array-convertible object. `dtype`: DTypeLike or None; inferred from input when omitted. |
| `jnp.concatenate` | `(arrays, axis=0, dtype=None)` | `arrays`: ndarray, JAX Array, or sequence of ArrayLike with matching shapes except along `axis`. `dtype`: DTypeLike or None; omitted dtype follows type promotion. |
| `jnp.stack` | `(arrays, axis=0, out=None, dtype=None)` | `arrays`: ndarray, JAX Array, or sequence of ArrayLike with matching shapes. `dtype`: DTypeLike or None; omitted dtype follows type promotion. |
| `jnp.clip` | `(arr=None, min=None, max=None, a=Deprecated, a_min=Deprecated, a_max=Deprecated)` | `arr`, `min`, `max`: ArrayLike or None; bounds must be broadcast-compatible with `arr`. |
| `jnp.zeros` | `(shape, dtype=None, device=None)` | `shape`: shape-like. `dtype`: DTypeLike or None; defaults to float32/float64 depending on `jax_enable_x64`. |
| `jnp.ones` | `(shape, dtype=None, device=None)` | `shape`: shape-like. `dtype`: DTypeLike or None; defaults to float32/float64 depending on `jax_enable_x64`. |
| `jnp.arange` | `(start, stop=None, step=None, dtype=None, device=None)` | `start`/`stop`: ArrayLike or DimSize. `step`: ArrayLike or None. `dtype`: DTypeLike or None; omitted dtype follows promotion of start/stop/step. |
| `jnp.interp` | `(x, xp, fp, left=None, right=None, period=None)` | `x`, `xp`, `fp`: ArrayLike; `xp` must be 1-D sorted and `fp.shape == xp.shape`. `left`/`right`: ArrayLike, 'extrapolate', or None. `period`: ArrayLike or None. |

**nonsmooth**

| Name | Signature | Inputs / dtype contract |
|---|---|---|
| `jnp.sign` | `(x)` | `x`: ArrayLike real or complex. Returns same shape and dtype as `x`. |
| `jnp.maximum` | `(*args, out=None, where=None)` | `x`, `y`: ArrayLike scalars/arrays; inputs must share shape or be broadcast-compatible. Result follows JAX type promotion and propagates NaN. |
| `jnp.minimum` | `(*args, out=None, where=None)` | `x`, `y`: ArrayLike scalars/arrays; inputs must share shape or be broadcast-compatible. Result follows JAX type promotion and propagates NaN. |
| `jnp.tanh` | `(x)` | `x`: ArrayLike numeric. Integers/bools are promoted to inexact dtype; complex inputs return complex. |

## Gotchas (curated — these are the ones that bite)

- `jax.config.update('jax_enable_x64', True)` must appear before any array is created. Without it every solve silently runs in float32 and stiff integrations lose accuracy.
- No Python control flow on traced values. `if failed: ...` fails to trace; use `jnp.where(failed, constants['error_loss'], loss_value)`.
- No in-place assignment. Use `arr.at[i].set(v)`, not `arr[i] = v`.
- NaN poisons reverse-mode autodiff even when multiplied by zero: the gradient of `jnp.where(mask, x, 0.0)` is NaN if `x` is NaN. Sanitise the data BEFORE the arithmetic: `safe = jnp.where(mask, dataset, 0.0)`, then divide. See `lib/LLM/reference/staggered_data.md`.
- Array shapes are static. Boolean-mask indexing (`x[mask]`) does not compile; multiply by a 0/1 mask and divide by `mask.sum()` instead.
