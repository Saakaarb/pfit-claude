Convert user_model.py to a JAX-jittable generated_script.py for the current session.

This file is PROCEDURE only. Every translation rule lives in
`lib/LLM/reference/jax_translation.md` — apply it from there, not from memory.

## Reference files (all required)

| File | Why |
|---|---|
| `lib/LLM/reference/cold_start.md` | the invariant: setup choices are made without the solution |
| `lib/LLM/reference/jax_translation.md` | the complete translation rules: structure, copied functions, both substitutions, the failure mask, SaveAt, output format |
| `lib/LLM/api/diffrax.md` | valid solver names, the `RESULTS` table, real signatures |
| `lib/LLM/api/jax.md` | available `jnp` functions and the tracing rules |
| `lib/utils/output_sample.py` | the template whose functions are copied verbatim |
| `lib/LLM/reference/user_model_contract.md` | what the source pseudocode means |
| `lib/LLM/reference/staggered_data.md` | ONLY if `_compute_loss_problem` masks NaNs |

If a digest is missing, or its header versions do not match the installed
packages, regenerate first:

```bash
./venv/bin/python3 tools/gen_api_context.py
```

## Steps

1. Ask the user for the session name if not given as an argument.

2. Read `sessions/<session_name>/inputs/user_input.xml` (for the parameter
   order, `MAX_STEPS` and `INTEGRATOR`),
   `sessions/<session_name>/generated/user_model.py`, and the reference files
   above.

3. Generate the script exactly as `jax_translation.md` specifies.

4. Write it to `sessions/<session_name>/generated/generated_script.py`.

5. Verify it imports, per the verification command in `jax_translation.md`. On
   failure, fix and retry up to 3 times.

6. Tell the user the outcome:
   - success: "generated_script.py created and verified. Run
     `./venv/bin/python3 fit_parameters.py <session_name>` to start optimization."
   - otherwise: show the error and the relevant section of the generated script.
