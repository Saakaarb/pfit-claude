"""Unit tests for lib/utils/yamlread.py.

YAMLReader is the single point where user intent enters the framework, and most
of its fields would otherwise be unexercised: the committed session fixtures
never set every optional key, so a parsing regression in any of them would go
unnoticed until a real fit silently used the wrong setting.

Two groups of tests here exist because of YAML specifically:

- the numeric-coercion tests, because PyYAML is YAML 1.1 and hands back `1e-7`
  as a *string* (a float literal needs a dot in the mantissa and a signed
  exponent). Every bound and tolerance in this project is written in that form.
- the bool-coercion tests, because unquoted `on`/`no`/`yes` become booleans, so
  a state variable with one of those names would arrive as True/False.
"""

import shutil
import textwrap

import pytest
import yaml

from lib.utils.yamlread import InputError, YAMLReader, read_input_file
from tests.conftest import FIXTURES, remove_setting, set_setting


def read(config_path) -> YAMLReader:
    return read_input_file(config_path)


def read_text(text: str) -> YAMLReader:
    reader = YAMLReader()
    reader.read(yaml.safe_load(textwrap.dedent(text)))
    return reader


@pytest.fixture
def decay_config(tmp_path):
    """A writable copy of the single-experiment fixture config."""
    dest = tmp_path / "user_input.yaml"
    shutil.copy(FIXTURES / "decay_session" / "inputs" / "user_input.yaml", dest)
    return dest


MINIMAL = """
    experiments:
      - data_file: d.csv
    model:
      trainable_parameters:
        - {name: k1, min_val: 0.01, max_val: 10.0, logscale: true}
      integrated_variables:
        - {name: A, init_val: 1.0}
    population_opt: {population_size: 20, num_iters: 5, processors: 2}
    gradient_opt:
      num_iters: 8
      stepsize_rtol: 1.0e-08
      stepsize_atol: 1.0e-10
      initial_timestep: 1.0e-04
      max_steps: 4096
"""


# ---------------------------------------------------------------------------
# model section
# ---------------------------------------------------------------------------

def test_parses_model_description(decay_config):
    r = read(decay_config)
    assert r.n_search_axes == 2
    assert r.trainable_parameter_names == ["k1", "k2"]
    assert r.min_axis_values == [0.01, 0.01]
    assert r.max_axis_values == [10.0, 10.0]
    assert r.axis_logscale == [1, 1]
    assert r.integrated_variable_names == ["A", "B"]
    assert r.integrated_variable_init_values == [1.0, 0.0]


def test_n_search_axes_follows_the_parameter_list_length():
    """The XML carried a separate declared count that could disagree with the
    list; in YAML the list length is the only source."""
    r = read_text(MINIMAL)
    assert r.n_search_axes == 1 == len(r.trainable_parameter_names)


def test_logscale_accepts_bools_and_the_y_n_spelling():
    for literal, expected in (("true", 1), ("false", 0), ("Y", 1), ("N", 0)):
        r = read_text(MINIMAL.replace("logscale: true", f"logscale: {literal}"))
        assert r.axis_logscale == [expected], literal


def test_logscale_rejects_anything_else():
    with pytest.raises(InputError, match="expected true or false"):
        read_text(MINIMAL.replace("logscale: true", "logscale: maybe"))


def test_unknown_key_inside_a_parameter_raises():
    with pytest.raises(InputError, match="unknown key"):
        read_text(MINIMAL.replace("min_val: 0.01", "minimum: 0.01"))


def test_missing_required_parameter_key_raises():
    with pytest.raises(InputError, match="required key 'max_val' is missing"):
        read_text(MINIMAL.replace("max_val: 10.0, ", ""))


def test_check_name_uniqueness_rejects_collisions():
    r = read_text(MINIMAL.replace("{name: A, init_val: 1.0}", "{name: k1, init_val: 1.0}"))
    with pytest.raises(ValueError, match="not unique"):
        r.check_name_uniqueness()


def test_check_name_uniqueness_passes_on_clean_input(decay_config):
    read(decay_config).check_name_uniqueness()  # must not raise


def test_fixed_parameters_are_optional_and_parsed():
    r = read_text(MINIMAL.replace(
        "      integrated_variables:",
        "      fixed_parameters:\n"
        "        - {name: c, value: 5.0}\n"
        "      integrated_variables:"))
    assert r.fixed_parameter_names == ["c"]
    assert r.fixed_parameter_values == [5.0]


# ---------------------------------------------------------------------------
# YAML 1.1 numeric coercion — the trap this format change introduces
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("literal", ["1e-7", "1.0e-07", "5E3", "1E+8", "0.0001"])
def test_exponent_literals_are_coerced_regardless_of_yaml_type(literal):
    """PyYAML returns `1e-7` and `5E3` as strings, not floats.

    The reader must coerce every numeric field explicitly; trusting the scalar
    type PyYAML infers would put a string into the solver's tolerances.
    """
    r = read_text(MINIMAL.replace("max_val: 10.0", f"max_val: {literal}"))
    assert r.max_axis_values == [float(literal)]


def test_a_non_numeric_bound_is_reported_with_its_location():
    with pytest.raises(InputError, match=r"trainable_parameters\[0\].max_val"):
        read_text(MINIMAL.replace("max_val: 10.0", "max_val: banana"))


def test_a_boolean_is_not_accepted_as_a_number():
    with pytest.raises(InputError, match="expected a number"):
        read_text(MINIMAL.replace("max_val: 10.0", "max_val: true"))


def test_max_steps_must_be_a_whole_number():
    with pytest.raises(InputError, match="expected an integer"):
        read_text(MINIMAL.replace("max_steps: 4096", "max_steps: 40.5"))


# ---------------------------------------------------------------------------
# YAML bool coercion of names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["on", "off", "yes", "no", "true"])
def test_a_yaml_bool_token_as_a_variable_name_is_rejected_clearly(name):
    """Unquoted on/off/yes/no arrive as booleans, so the name the user typed is
    not the name that reaches the reader. Fail with an explanation."""
    with pytest.raises(InputError, match="YAML boolean"):
        read_text(MINIMAL.replace("{name: A, init_val: 1.0}",
                                  "{name: %s, init_val: 1.0}" % name))


def test_a_name_that_is_not_an_identifier_is_rejected():
    with pytest.raises(InputError, match="valid Python identifier"):
        read_text(MINIMAL.replace("{name: A,", '{name: "my var",'))


# ---------------------------------------------------------------------------
# optimizer settings
# ---------------------------------------------------------------------------

def test_population_settings(decay_config):
    r = read(decay_config)
    assert r.population_size == 20
    assert r.n_iters_pop == 5
    assert r.processors == 2
    assert r.pop_stepsize_rtol == [1e-5, 1e-5]
    assert r.pop_stepsize_atol == [1e-7, 1e-7]


def test_gradient_settings(decay_config):
    r = read(decay_config)
    assert r.n_iters_grad == 8
    assert r.stepsize_rtol == [1e-8, 1e-8]
    assert r.stepsize_atol == [1e-10, 1e-10]
    assert r.init_timestep == 1e-4
    assert r.max_steps == 4096
    assert r.integrator == "Dopri5"
    assert r.gradient_optimizer == "lbfgs"
    assert r.init_value_lr == 1e-2
    assert r.end_value_lr == 1e-4
    assert r.transition_steps_lr == 50
    assert r.decay_rate_lr == 0.9


@pytest.mark.parametrize("value", ["PSO", "DE"])
def test_algorithm_is_parsed(decay_config, value):
    set_setting(decay_config, "population_opt", "algorithm", value)
    assert read(decay_config).algorithm == value


def test_random_seed_is_parsed(decay_config):
    set_setting(decay_config, "population_opt", "random_seed", 7)
    assert read(decay_config).random_seed == 7


def test_gradient_optimizer_is_lowercased(decay_config):
    """The NODE class compares against lowercase names, so casing must not matter."""
    set_setting(decay_config, "gradient_opt", "gradient_optimizer", "ADAM")
    assert read(decay_config).gradient_optimizer == "adam"


def test_initial_time_is_parsed(decay_config):
    set_setting(decay_config, "gradient_opt", "initial_time", 0.5)
    assert read(decay_config).init_time == 0.5


def test_unknown_population_key_raises(decay_config):
    set_setting(decay_config, "population_opt", "num_particles", 50)
    with pytest.raises(InputError, match="unknown key"):
        read(decay_config)


def test_unknown_gradient_key_also_raises(decay_config):
    """The XML reader silently DROPPED unknown gradient keys, so a typo'd
    max_steps was applied to nothing and never reported. Strictness is now
    uniform across sections."""
    set_setting(decay_config, "gradient_opt", "maxsteps", 999)
    with pytest.raises(InputError, match="unknown key"):
        read(decay_config)


def test_unknown_top_level_section_raises(decay_config):
    text = decay_config.read_text() + "\nplotting_info:\n  write_results: true\n"
    decay_config.write_text(text)
    with pytest.raises(InputError, match="unknown key"):
        read(decay_config)


def test_missing_required_section_raises():
    config = yaml.safe_load(textwrap.dedent(MINIMAL))
    del config["gradient_opt"]
    reader = YAMLReader()
    with pytest.raises(InputError, match="required key 'gradient_opt' is missing"):
        reader.read(config)


def test_defaults_when_optional_settings_are_absent(decay_config):
    for key in ("integrator", "gradient_optimizer"):
        remove_setting(decay_config, "gradient_opt", key)
    for key in ("stepsize_rtol", "stepsize_atol"):
        remove_setting(decay_config, "population_opt", key)
    r = read(decay_config)
    assert r.integrator == "Kvaerno5"
    assert r.gradient_optimizer == "lbfgs"
    assert r.algorithm == "PSO"
    assert r.random_seed is None
    assert r.pop_stepsize_rtol is None
    assert r.pop_stepsize_atol is None


def test_write_results_flag(decay_config):
    assert read(decay_config).write_results is True
    set_setting(decay_config, "output", "write_results", False)
    assert read(decay_config).write_results is False


def test_write_results_defaults_to_false_without_an_output_section():
    assert read_text(MINIMAL).write_results is False


# ---------------------------------------------------------------------------
# tolerances
# ---------------------------------------------------------------------------

def test_a_scalar_tolerance_broadcasts_to_every_variable(decay_config):
    """Two state variables, one value given."""
    set_setting(decay_config, "gradient_opt", "stepsize_rtol", 1.0e-8)
    assert read(decay_config).stepsize_rtol == [1e-8, 1e-8]


def test_an_explicit_tolerance_list_is_kept_per_variable(decay_config):
    set_setting(decay_config, "gradient_opt", "stepsize_rtol", [1.0e-8, 1.0e-6])
    assert read(decay_config).stepsize_rtol == [1e-8, 1e-6]


def test_a_wrong_length_tolerance_list_is_rejected(decay_config):
    """The XML format documented this requirement but nothing enforced it."""
    set_setting(decay_config, "gradient_opt", "stepsize_rtol", [1.0e-8, 1.0e-8, 1.0e-8])
    with pytest.raises(InputError, match="got 3 value\\(s\\) but the model has 2"):
        read(decay_config)


# ---------------------------------------------------------------------------
# experiments and initial conditions
# ---------------------------------------------------------------------------

def test_single_experiment(decay_config):
    r = read(decay_config)
    assert len(r.experiments) == 1
    assert r.experiments[0]["filename"] == "decay_data.csv"
    assert r.experiments[0]["ic_overrides"] == {}
    assert r.filename_data == "decay_data.csv"
    assert r.get_y0(0) == [1.0, 0.0]


def test_multi_experiment_initial_condition_overrides():
    """Per-experiment initial_conditions must override only the named variables."""
    r = read(FIXTURES / "decay_multiexp" / "inputs" / "user_input.yaml")
    assert len(r.experiments) == 2
    assert [e["filename"] for e in r.experiments] == ["decay_run_A.csv", "decay_run_B.csv"]

    assert r.experiments[0]["ic_overrides"] == {}
    assert r.get_y0(0) == [1.0, 0.0]          # falls back to the global init_vals

    assert r.experiments[1]["ic_overrides"] == {"A": 2.0, "B": 0.5}
    assert r.get_y0(1) == [2.0, 0.5]          # overridden


def test_partial_initial_condition_override(tmp_path):
    """A variable absent from initial_conditions keeps its global init_val."""
    config = tmp_path / "user_input.yaml"
    shutil.copy(FIXTURES / "decay_multiexp" / "inputs" / "user_input.yaml", config)
    config.write_text(config.read_text().replace("      B: 0.5\n", ""))

    r = read(config)
    assert r.experiments[1]["ic_overrides"] == {"A": 2.0}
    assert r.get_y0(1) == [2.0, 0.0]  # A overridden, B from the global value


def test_unknown_initial_condition_name_is_rejected_at_parse_time(tmp_path):
    """The XML reader accepted this and only failed later, inside get_y0."""
    config = tmp_path / "user_input.yaml"
    shutil.copy(FIXTURES / "decay_multiexp" / "inputs" / "user_input.yaml", config)
    config.write_text(config.read_text().replace("      A: 2.0", "      typo: 2.0"))

    with pytest.raises(InputError, match="not an integrated variable"):
        read(config)


def test_get_y0_does_not_mutate_global_init_values():
    """get_y0 must return a copy; otherwise experiment 2's ICs would leak into
    experiment 1 on the next call."""
    r = read(FIXTURES / "decay_multiexp" / "inputs" / "user_input.yaml")
    r.get_y0(1)
    assert r.integrated_variable_init_values == [1.0, 0.0]
    assert r.get_y0(0) == [1.0, 0.0]


def test_at_least_one_experiment_is_required():
    config = yaml.safe_load(textwrap.dedent(MINIMAL))
    config["experiments"] = []
    with pytest.raises(InputError, match="at least one experiment"):
        YAMLReader().read(config)


# ---------------------------------------------------------------------------
# paths and file-level errors
# ---------------------------------------------------------------------------

def test_paths_section_overrides_directory_names():
    r = read_text(MINIMAL + """
    paths:
      user_input_dir: in
      generated_dir: gen
      output_dir: out
    """)
    assert (r.user_input_dirname, r.generated_dirname, r.output_dirname) == ("in", "gen", "out")


def test_malformed_yaml_names_the_file(tmp_path):
    config = tmp_path / "user_input.yaml"
    config.write_text("experiments: [unclosed\n")
    with pytest.raises(InputError, match="not valid YAML"):
        read(config)


def test_errors_are_prefixed_with_the_file_path(tmp_path):
    config = tmp_path / "user_input.yaml"
    config.write_text("experiments: []\n")
    with pytest.raises(InputError, match=str(config)):
        read(config)
