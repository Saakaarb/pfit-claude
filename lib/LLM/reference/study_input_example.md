---
topic: User-facing study input example for building a fitting session
consumed_by: [pfit-new]
generated: false
owns: >
  A general prose-first example with two raw datasets, experiment-specific
  initial conditions and forcing histories, canonical fitting CSV formatting,
  and the targeted prompts used to complete the agent's internal study picture.
---

# Study input example

The user should not have to build framework tables. They should provide raw CSV
files plus a plain-language note or paper. The agent's job is to build a full
mental picture of the run: what gets integrated for each experiment, what differs
between experiments, how every useful data column enters the RHS or loss, and
which files must be generated to run and inspect the fit.

## Raw data files

Initial raw files may differ in row count, time grid, column order, extra
columns, and column names. They may also omit a measurement at some time points.
The agent must not modify data files or write derived data files. After the
agent infers and reviews the mapping, the user creates framework-ready CSVs with
the canonical layout.

Minimum raw CSV rules:

- comma-delimited text;
- one time column per file;
- numeric data after the optional header row;
- monotonic increasing time values;
- no hidden Excel formatting;
- blanks only when the missing measurement is intentional; those blanks become
  NaN and require a nan-safe loss.

For a multi-experiment fit, the canonical fitting copies must use the same
column meaning and order. Row counts, sampling times, initial conditions, and
forcing values may differ between experiments. If an experiment lacks a fitted
measurement at some time point, keep the canonical column and put a blank/NaN in
that row.

Example raw file 1: `pulse_low_raw.csv`

```csv
time_min,dose_rate_uM_per_min,fluorescence,fluorescence_sd,temperature_C
0,0.0,100.0,4.0,37
1,1.0,,4.2,37
2,1.0,113.5,4.4,37
4,0.0,,4.1,37
6,0.0,104.5,4.0,37
```

Example raw file 2: `washout_raw.csv`

```csv
t,signal_sd,signal
0,5.0,130.0
0.5,5.1,126.0
1.0,5.5,
2.5,5.2,118.0
5.0,4.8,110.0
8.0,4.7,104.0
```

These initial files are intentionally different. The first file has missing
fluorescence measurements at some forcing sample times. The second file has a
different time header, different column order, more rows, and no dose-rate
column because it is a washout experiment with no input dose.

## What the user writes

The note can be simple, but it should use predictable subheadings so the agent
can extract the model, parameters, data mapping, and loss without asking the
user to fill out framework tables.

```markdown
# Drug response fitting problem

## User model

Use `pulse_low_raw.csv` for the low-dose pulse and `washout_raw.csv` for the
washout experiment where the dose input is zero the whole time.

The states are D and R.

Initial conditions:
- low-dose pulse: D = 0, R = 0
- washout: D = 0, R = 0.3

dD/dt = dose_rate(t) - k_elim * D
dR/dt = k_on * D * (1 - R) - k_off * R

The measured signal is:

signal = baseline + gain * R

## Trainable parameters

Fit these parameters:

k_elim: 0.01 to 10, log scale
k_on: 0.001 to 10, log scale
k_off: 0.001 to 10, log scale
baseline: 50 to 150, linear scale
gain: 1 to 200, linear scale

## Fixed parameters

There are no fixed parameters.

## Dataset columns and forcing

In the low-dose file, dose_rate_uM_per_min is dose_rate(t), fluorescence is the
measured signal, and fluorescence_sd is its uncertainty.

In the washout file, there is no dose column because dose_rate(t) is zero.
signal is the measured signal and signal_sd is its uncertainty.

Ignore temperature_C.

## Loss function

For each dataset, calculate the model output as:

signal_model = baseline + gain * R

Compare signal_model to that dataset's measured signal column. Divide each
residual by that dataset's signal uncertainty column.

Use one combined loss across both datasets:

loss = sqrt(mean(all finite residuals from both datasets squared))

Rows where the measured signal is blank should be ignored in the loss, but kept
in the CSV so dose_rate(t) is still defined at those times.
```

This is enough. The user does not need to number columns, explain what every
parameter means, write an identifiability map, or write YAML.

## What the agent infers

The complete internal picture is:

```text
Both experiments integrate the same ODE system with states D and R.

Experiment 1 starts from D=0, R=0.
Experiment 2 starts from D=0, R=0.3.

dose_rate(t) is experiment-specific and comes from a CSV column:
- pulse_low_raw.csv: dose_rate_uM_per_min
- washout_raw.csv: no raw column, so fill canonical dose_rate with 0.0

The fitted model output is the derived observable:
signal = baseline + gain * R

The loss compares signal to each experiment's measured signal column:
- pulse_low_raw.csv: fluorescence
- washout_raw.csv: signal

Each residual is divided by that experiment's uncertainty column:
- pulse_low_raw.csv: fluorescence_sd
- washout_raw.csv: signal_sd

temperature_C is ignored.
```

The canonical fitting layout can then be:

```csv
time,dose_rate,signal,signal_sd
```

The user writes framework-ready files such as:

```text
sessions/drug_response/inputs/pulse_low.csv
sessions/drug_response/inputs/washout.csv
```

The original raw uploads remain unchanged.

## Making the data framework-ready

After the mapping is reviewed, the user converts the raw files into
framework-ready CSVs with the same column order. Missing measurements stay as
blanks/NaNs so the time-varying forcing can still be sampled at those rows
without pretending a measurement exists.

First define the canonical columns:

```text
time      <- the experiment time grid
dose_rate <- the forcing used as dose_rate(t)
signal    <- the measured value compared to baseline + gain * R
signal_sd <- the uncertainty used to weight the signal residual
```

Then map each raw file into those columns:

```text
pulse_low_raw.csv:
  time      <- time_min
  dose_rate <- dose_rate_uM_per_min
  signal    <- fluorescence
  signal_sd <- fluorescence_sd
  drop      <- temperature_C

washout_raw.csv:
  time      <- t
  dose_rate <- fill with 0.0 because the note says this is a zero-input washout
  signal    <- signal
  signal_sd <- signal_sd
```

Concrete row conversions:

```text
pulse_low_raw row:
  1,1.0,,4.2,37
becomes:
  1,1.0,,4.2

washout_raw row:
  0.5,5.1,126.0
becomes:
  0.5,0.0,126.0,5.1

washout_raw row with missing signal:
  1.0,5.5,
becomes:
  1.0,0.0,,5.5
```

Canonical file 1: `pulse_low.csv`

```csv
time,dose_rate,signal,signal_sd
0,0.0,100.0,4.0
1,1.0,,4.2
2,1.0,113.5,4.4
4,0.0,,4.1
6,0.0,104.5,4.0
```

Canonical file 2: `washout.csv`

```csv
time,dose_rate,signal,signal_sd
0,0.0,130.0,5.0
0.5,0.0,126.0,5.1
1.0,0.0,,5.5
2.5,0.0,118.0,5.2
5.0,0.0,110.0,4.8
8.0,0.0,104.0,4.7
```

Because the canonical files contain blanks in a fitted observable column, the loss
must use the nan-safe pattern from `staggered_data.md`: create the finite mask,
replace NaNs before arithmetic, and average only measured residuals. Filling a
missing forcing column is different: it is allowed only when the user/source
states the forcing value, as with the zero-input washout above.

## Targeted prompts

If the note were missing details, the agent should ask only about those details.

```text
I found two signal columns: `fluorescence` in `pulse_low_raw.csv` and
`signal` in `washout_raw.csv`. Should both map to the model observable
`signal = baseline + gain * R`?
```

```text
`washout_raw.csv` starts with R = 0.3 in your note, while the low-dose file
starts with R = 0. Should I treat those as experiment-specific initial
conditions?
```

```text
The low-dose file has `temperature_C`. It does not appear in the equations or
loss. Should I ignore it?
```

```text
The equations use `dose_rate(t)`, but `washout_raw.csv` has no dose column.
Should I fill the canonical `dose_rate` column with 0.0 for that experiment?
```

These prompts come from gaps in the internal picture. They are not a fixed form.

## Files generated for the current framework

The agent compiles the reviewed picture into `user_input.yaml` and
`user_model.py`.

The config uses the canonical files and common column layout:

```yaml
experiments:
  - data_file: pulse_low.csv
    initial_conditions:
      D: 0.0
      R: 0.0
    columns:
      - {name: time}
      - {name: dose_rate}
      - {name: signal, observes: signal}
      - {name: signal_sd, uncertainty_of: signal}
  - data_file: washout.csv
    initial_conditions:
      D: 0.0
      R: 0.3
    columns:
      - {name: time}
      - {name: dose_rate}
      - {name: signal, observes: signal}
      - {name: signal_sd, uncertainty_of: signal}

model:
  trainable_parameters:
    - {name: k_elim, min_val: 0.01, max_val: 10.0, logscale: true}
    - {name: k_on, min_val: 0.001, max_val: 10.0, logscale: true}
    - {name: k_off, min_val: 0.001, max_val: 10.0, logscale: true}
    - {name: baseline, min_val: 50.0, max_val: 150.0, logscale: false}
    - {name: gain, min_val: 1.0, max_val: 200.0, logscale: false}
  observables:
    - {name: signal}
  integrated_variables:
    - {name: D, init_val: 0.0}
    - {name: R, init_val: 0.0}
```

The model code uses the canonical column positions. Inside `user_model.py`,
`dataset` excludes the time column, so `dataset[:, 0]` is `dose_rate`,
`dataset[:, 1]` is `signal`, and `dataset[:, 2]` is `signal_sd`.

```python
def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    D, R = y
    k_elim = trainable_parameters["k_elim"]
    k_on = trainable_parameters["k_on"]
    k_off = trainable_parameters["k_off"]

    dose_rate = np.interp(t, t_eval, dataset[:, 0])

    dD_dt = dose_rate - k_elim * D
    dR_dt = k_on * D * (1.0 - R) - k_off * R
    return [dD_dt, dR_dt]


def _observables(solution, trainable_parameters, fixed_parameters):
    baseline = trainable_parameters["baseline"]
    gain = trainable_parameters["gain"]
    R = solution[:, 1]
    return {"signal": baseline + gain * R}


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    signal = _observables(solution, trainable_parameters, fixed_parameters)["signal"]
    measured = dataset[:, 1]
    sigma = dataset[:, 2]

    mask = ~np.isnan(measured)
    measured_safe = np.where(mask, measured, 0.0)
    sigma_safe = np.where(mask, sigma, 1.0)
    residual = np.where(mask, (signal - measured_safe) / sigma_safe, 0.0)
    return np.sqrt(np.sum(np.square(residual)) / np.sum(mask))
```

## Starting the workflow

The user starts with one command or button:

```text
/pfit-new drug_response
```

The agent gathers the writeup and CSVs, builds the full mental picture, asks
only the targeted prompts needed to complete it, waits for the user-prepared
framework-ready CSVs when needed, writes the session files, and then continues
with:

```text
/pfit-check drug_response
/pfit-jax drug_response
/pfit-run drug_response
```
