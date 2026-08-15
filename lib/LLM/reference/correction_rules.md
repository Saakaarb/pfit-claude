---
topic: How to auto-correct user_input.yaml and user_model.py from a validation report
consumed_by: [pfit-check]
generated: false
owns: The minimal-change correction policy and the per-error fix rules.
---

# Correction rules

Applied after validation (`validation_rules.md`) produces a report. You are an
assistant code editor making **minimal** changes to the existing config and
pseudocode to address the reported points. The `.py` file is pseudocode that
will be translated to JAX later — do not treat it as JAX code.

If no changes are required, make none.

## Policy

1. **Minimal changes.** Make as few edits as possible to fix as many critical
   errors as possible.
2. **Do not change the logic** of the code unless it is fundamentally wrong and
   was flagged in the report.
3. **Never reinterpret column indices.** Use them exactly as the user provided
   them; make no assumptions about what each dataset column means.
4. If an error cannot be fixed safely, leave it and tell the user.
5. Fix anything that would either error out or cause poor convergence — read
   both the Critical Errors and the Warnings sections.

## Policy on the dataset

**Never edit a dataset CSV.** It is the measurement, and the one artifact the
user owns outright. Every Dataset check (D1-D10 in `validation_rules.md`) is
therefore reported, never auto-corrected — even when the fix looks mechanical.
Stripping a trailing delimiter or dropping a duplicate timestamp silently
changes what the fit is being scored against.

The corrigible half of a dataset failure is always on the config/model side:

- **D6** (`t_eval[0] < initial_time`): the config is wrong far more often than
  the data. Report both numbers and propose removing or lowering `initial_time`;
  apply only on confirmation, since a deliberate offset is valid.
- **D8** (a literal column index the CSV does not have): flag it against the
  actual column count. Do NOT renumber the index — per policy 3, column meanings
  are the user's.
- **D3x** (NaN present, loss not nan-safe): flag it and point at
  `staggered_data.md`. Converting the loss to nan-safe reductions changes what is
  being fitted, so it is never automatic.

## Specific fixes

- **`initial_conditions/VAR` with an unmatched `NAME`:** remove that `VAR` block
  entirely. Do not guess a corrected name — there is no safe automatic fix.
- **Missing `FILENAME`/`data_file`:** flag it; do NOT invent a filename.
- **Missing `logscale: true` on a wide-range parameter:** set it.

## Loop

Validate -> correct -> re-validate, up to 3 iterations, stopping as soon as
there are no critical errors.
