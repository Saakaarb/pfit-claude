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

## Specific fixes

- **`initial_conditions/VAR` with an unmatched `NAME`:** remove that `VAR` block
  entirely. Do not guess a corrected name — there is no safe automatic fix.
- **Missing `FILENAME`/`data_file`:** flag it; do NOT invent a filename.
- **Missing `logscale: true` on a wide-range parameter:** set it.

## Loop

Validate -> correct -> re-validate, up to 3 iterations, stopping as soon as
there are no critical errors.
