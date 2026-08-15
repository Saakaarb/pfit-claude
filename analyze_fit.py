"""
Stand-alone post-processing of a completed pfit-claude session: compute the
sloppiness / identifiability diagnostics of Hass et al. (2019) for an existing
fit (uses outputs/final_design_point.csv).

The same analysis runs automatically at the end of every gradient-based fit
(see lib/utils/sloppiness.py, wired into helper_functions.py). This script just
lets you (re)run it on any session after the fact.

Usage:
    venv/bin/python3 analyze_fit.py <session_name>
"""
import os
import sys
import importlib.util
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.utils.helper_functions import get_input_reader, CreatedClass
from lib.utils.sloppiness import run_sloppiness_analysis


def resolve_session_dir(argv) -> Path:
    root = Path("sessions")
    if len(argv) == 2:
        arg = Path(argv[1])
        session = arg if arg.is_dir() else root / arg
        if not session.is_dir():
            raise ValueError(f"Session directory {session} does not exist")
        return session
    dirs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda d: d.stat().st_ctime, reverse=True)
    if not dirs:
        raise ValueError("No sessions found and none provided")
    return dirs[0]


def main():
    session = resolve_session_dir(sys.argv)
    print(f"Analysing session: {session.name}")

    # load the session's generated script
    gen_path = session / "generated" / "generated_script.py"
    spec = importlib.util.spec_from_file_location("generated_script", gen_path)
    gen = importlib.util.module_from_spec(spec)
    sys.modules["generated_script"] = gen
    spec.loader.exec_module(gen)

    # rebuild the framework problem object
    input_reader = get_input_reader(str(session / "inputs" / "user_input.yaml"))
    input_reader.check_name_uniqueness()
    input_reader.output_dir = session / "outputs"
    experiments = []
    for exp in input_reader.experiments:
        dpath = session / input_reader.user_input_dirname / exp["filename"]
        with open(dpath, "r", encoding="utf-8-sig") as f:
            data = np.genfromtxt(f, dtype=float, delimiter=",")
        experiments.append({"t_eval": data[:, 0], "dataset": data[:, 1:],
                            "y0": jnp.array(input_reader.get_y0(len(experiments)))})
    prob = CreatedClass(experiments, input_reader, gen._compute_loss_problem, gen._write_problem_result)

    # search-axis bounds (log10 of physical bounds for logscale params) + logscale flags
    is_log = list(input_reader.axis_logscale)
    lo, hi = [], []
    for i, lg in enumerate(is_log):
        mn, mx = input_reader.min_axis_values[i], input_reader.max_axis_values[i]
        lo.append(float(np.log10(mn)) if lg else float(mn))
        hi.append(float(np.log10(mx)) if lg else float(mx))
    prob.set_min_limit(lo); prob.set_max_limit(hi); prob.set_is_logscale(is_log)
    lo, hi = np.array(lo), np.array(hi)

    # best-fit physical params -> scaled [-1,1] search coordinate
    theta = np.atleast_1d(np.loadtxt(session / "outputs" / "final_design_point.csv"))
    u = np.where(np.array(is_log).astype(bool), np.log10(theta), theta)
    scaled_best = 2.0 * (u - lo) / (hi - lo) - 1.0

    run_sloppiness_analysis(gen._compute_loss_problem, prob.constants_list, scaled_best,
                            input_reader.trainable_parameter_names, session / "outputs")


if __name__ == "__main__":
    main()
