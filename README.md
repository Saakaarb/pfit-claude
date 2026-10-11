# pfit remote agent

`pfit` fits unknown parameters in ordinary differential equation (ODE) models
to time-series data. A remote coding agent prepares and validates each fitting
session; the numerical fit itself runs locally with JAX, Diffrax, and either
differential evolution or particle swarm optimization followed by gradient
refinement.

The supported workflow is:

```text
pfit new -> pfit check -> pfit jax -> pfit run -> pfit diagnose
```

This repository supports both **Claude Code** and **OpenAI Codex**. It does not
call an LLM API from the Python fitting process: use the subscription or remote
agent environment supplied by your chosen coding agent.

## Requirements

- macOS or Linux
- Python 3.12
- Claude Code or OpenAI Codex
- enough local CPU and memory for the chosen population size

Model descriptions, source papers, and data opened by a remote agent may be
sent to that agent's provider. Do not use restricted data unless your provider
and account are approved for it.

## Install

Clone the repository, enter it, and create the environment expected by the
agent procedures:

```bash
python3.12 -m venv venv
./venv/bin/python3 -m pip install --upgrade pip
./venv/bin/python3 -m pip install -r requirements.txt
./venv/bin/python3 -m pytest -q
```

Keep the repository root as the agent's working directory. Do not commit API
keys, access tokens, source papers containing restricted material, or private
datasets.

## Quick start

Place one or more CSV files under `sessions/<name>/inputs/`, then describe the
equations, fitted parameters, parameter scales, initial conditions, and the
meaning of each data column to the agent. A paper or local write-up can supply
that description.

In Claude Code, invoke the slash commands:

```text
/pfit-new my_study
/pfit-check my_study
/pfit-jax my_study
/pfit-run my_study
/pfit-diagnose my_study <run-id>
```

In Codex, name the corresponding repository skill:

```text
$pfit-new my_study
$pfit-check my_study
$pfit-jax my_study
$pfit-run my_study
$pfit-diagnose my_study <run-id>
```

You can also ask naturally, for example, “Use pfit-new to create the `my_study`
session from these equations and CSV files.” The agent will ask only for
scientific information it cannot infer safely. In particular, it will not
invent equations, datasets, initial conditions, or unsupported parameter
ranges.

## Session layout

```text
sessions/<name>/
  inputs/
    user_input.yaml
    data.csv
  generated/
    user_model.py
    generated_script.py
    user_input_check.txt
  outputs/
    <run-id>/
```

`pfit-new` creates the YAML configuration and Python model together.
`pfit-check` validates and, with confirmation, corrects them. `pfit-jax`
performs the agent translation to JAX and deterministically verifies and stamps
the result. `pfit-run` refuses stale or incomplete generated code before
starting the local optimization. `pfit-diagnose` examines one immutable run and
writes an evidence-based report.

All accepted inputs are YAML and CSV. XML is not supported.

## Running without an agent

After an agent has created and verified `generated/generated_script.py`, the
numerical entry points can be run directly:

```bash
./venv/bin/python3 tools/check_ready.py my_study --mode full
./venv/bin/python3 fit_parameters.py my_study
```

For gradient-only refinement from a previous run:

```bash
./venv/bin/python3 tools/check_ready.py my_study --mode gradient-only
./venv/bin/python3 fit_gradient_only.py my_study --seed-run <run-id>
```

The JAX translation is intentionally an agent step; do not treat it as a
mechanical source-to-source conversion. The stamp and readiness checks ensure
that the generated script still matches the model and YAML from which it was
created.

## Reproducibility and support

Each new fit gets its own output directory with input snapshots and a manifest.
Earlier runs are preserved. Start troubleshooting with the failing command's
report, then run:

```bash
./venv/bin/python3 tools/check_dataset.py my_study
./venv/bin/python3 tools/check_ready.py my_study --mode full
```

When reporting a problem, include the command, Python version, platform, run ID,
and the relevant report or log. Remove sensitive data before sharing it.

## Development

Run the fast test suite from the repository root:

```bash
./venv/bin/python3 -m pytest -m "not slow" -q
```

The agent procedures are in `.claude/commands/`; Codex skills deliberately
delegate to those files so both integrations use one scientific contract.

Licensed under Apache License 2.0. See [LICENSE](LICENSE).
