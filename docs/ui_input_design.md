# Study Input UI Design

The fitting workflow should ask users for a study description, not raw
framework files. The user supplies a paper, notes, or short writeup plus one or
more CSV datasets. The tool inspects those inputs, drafts a structured study
spec, and asks only for missing or ambiguous information.

The user should not be asked to create framework tables. The preferred input is
a plain-language note with equations, initial conditions, parameter bounds, and
short sentences such as "fit fluorescence", "use fluorescence_sd as its
uncertainty", or "dose_rate_uM_per_min is the input dose over time." The tool
uses the CSV header and equation symbols to infer the structured map.

The central internal object is a data-to-model map. Every CSV column must end up
with one role: time, observed state, observed derived quantity, forcing,
uncertainty, or ignore. This map should be presented as a review/confirmation
surface with accept/change controls, not as something the user authors from
scratch.

Every new fit should start by asking whether the agent has the full mental
picture of the run. This means the agent can explain the whole path from the
user's raw inputs to the files and plots produced by the workflow:

1. What ODE system is integrated.
2. Which symbols are states, fitted parameters, fixed parameters, forcings,
   time, or allowed math functions.
3. What differs between experiments: CSV file, time grid, forcing history,
   initial conditions, or measured observables.
4. How every useful CSV column enters the problem: time grid, RHS forcing,
   fitted measurement, uncertainty weight, or ignored metadata.
5. Which model quantity each fitted data column measures, including derived
   observables that are functions of the states.
6. What arithmetic defines the loss and which columns are excluded from it.
7. Which parameters the optimizer can change, with bounds and linear/log scale.
8. Which session files will be written and what will be plotted after fitting.

If this picture is complete, the tool should show a short review and proceed.
If not, the missing pieces become targeted prompts. The prompt is driven by the
unresolved symbol, column, or setting, not by a fixed questionnaire.

Completeness should be enforced by symbol checks. Every equation symbol must
resolve to a state, parameter, fixed parameter, forcing, time, or allowed math
function. Every fitted column must map to a state or derived observable. Every
forcing must have a formula, constant, or CSV source. Every fitted parameter
must have bounds and a linear/log scale. Every state must have an initial
condition or an explicit experiment-level override.

The user-facing flow should be:

1. Upload writeup/paper and CSV files.
2. Auto-draft states, equations, parameters, forcings, observables, column
   roles, and loss.
3. Ask targeted questions for only unresolved items.
4. Show the inferred map for review.
5. Compile the completed study spec into `user_input.yaml` and `user_model.py`.
6. Run the existing `/pfit-check`, `/pfit-jax`, and `/pfit-run` workflow.

The UI should call the map an "Observation Map" or "Data-to-Model Map" rather
than an "identifiability map." Identifiability is better presented later as a
diagnostic result derived from the completed map and fitted model.
