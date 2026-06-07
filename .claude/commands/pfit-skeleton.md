Generate a user_model.py skeleton for the current session from its user_input.xml.

## Steps

1. Ask the user for the session name if not provided as an argument (e.g. `vanderpol_session`). The session directory is `sessions/<session_name>/`.

2. Read the session's XML: `sessions/<session_name>/inputs/user_input.xml`

3. Read these reference files to understand what to generate:
   - `lib/LLM/user_model_generation_instructions.txt` — generation rules
   - `lib/utils/user_model_sample_unpopulated.py` — skeleton template
   - `lib/utils/user_model_sample_populated.py` — populated example (Robertson)
   - `lib/utils/user_input_sample.xml` — example XML (Robertson)

4. Following the rules in `user_model_generation_instructions.txt` and using the Robertson example as a reference, generate a `user_model.py` skeleton populated with:
   - The correct trainable parameter names (from `TRAINABLE_PARAMETER_DESCRIPTION`)
   - The correct fixed parameter names (from `FIXED_PARAM_DESCRIPTION`)
   - The correct integrated variable names and initial conditions (from `INTEGRATED_SYSTEM_DESCRIPTION`)
   - Comments indicating the ordering of trainable parameters in the vector
   - All three function stubs: `user_defined_system`, `_compute_loss_problem`, `writeout_description`

5. Create the `sessions/<session_name>/generated/` directory if it doesn't exist.

6. Write the skeleton to `sessions/<session_name>/generated/user_model.py`.

7. If the XML contains multiple `<EXPERIMENT>` blocks, add a comment near the top of each function (below the argument list) stating: `# dataset and t_eval represent ONE experiment's data; the framework calls this function once per experiment`.

8. Tell the user what was generated and remind them to fill in the ODE logic in `user_defined_system`, the loss computation in `_compute_loss_problem`, and the writeout in `writeout_description` before running `/pfit-check`.

9. At the end, VERIFY your implementation by cross checking against the input XML again

## Rules (from user_model_generation_instructions.txt)
- Do NOT leave any function empty — include stubs with comments showing what to fill in
- Add `import numpy as np` at the top
- Do NOT add any boilerplate text; the file should be pure Python
- Preserve all comments from the template
- For multi-experiment XMLs: `dataset` and `t_eval` always represent a single experiment; do NOT generate loops over multiple datasets inside any of the three functions
