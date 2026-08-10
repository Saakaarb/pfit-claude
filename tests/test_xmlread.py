"""Unit tests for lib/utils/xmlread.py.

XMLReader is the single point where user intent enters the framework, and most
of its fields were previously unexercised: the committed session fixtures never
set INTEGRATOR, GRADIENT_OPTIMIZER, ALGORITHM, RANDOM_SEED, POP_STEPSIZE_* or
per-experiment INITIAL_CONDITIONS, so a parsing regression in any of them would
have gone unnoticed until a real fit silently used the wrong setting.
"""

import xml.etree.ElementTree as ET

import pytest

from lib.utils.xmlread import XMLReader
from tests.conftest import FIXTURES, set_xml_setting, remove_xml_setting


def read(xml_path) -> XMLReader:
    reader = XMLReader()
    reader.read_XML(ET.parse(xml_path).getroot())
    return reader


@pytest.fixture
def decay_xml(tmp_path):
    """A writable copy of the single-experiment fixture XML."""
    import shutil

    dest = tmp_path / "user_input.xml"
    shutil.copy(FIXTURES / "decay_session" / "inputs" / "user_input.xml", dest)
    return dest


# ---------------------------------------------------------------------------
# model section
# ---------------------------------------------------------------------------

def test_parses_model_description(decay_xml):
    r = read(decay_xml)
    assert r.n_search_axes == 2
    assert r.trainable_parameter_names == ["k1", "k2"]
    assert r.min_axis_values == [0.01, 0.01]
    assert r.max_axis_values == [10.0, 10.0]
    assert r.axis_logscale == [1, 1]
    assert r.integrated_variable_names == ["A", "B"]
    assert r.integrated_variable_init_values == [1.0, 0.0]


def test_logscale_flag_must_be_y_or_n(decay_xml):
    text = decay_xml.read_text().replace("LOGSCALE = Y", "LOGSCALE = MAYBE", 1)
    decay_xml.write_text(text)
    with pytest.raises(ValueError, match="Unknown logscale value"):
        read(decay_xml)


def test_unknown_key_inside_param_raises(decay_xml):
    text = decay_xml.read_text().replace(
        "<P> MIN_VAL = 0.01 </P>", "<P> MINIMUM = 0.01 </P>", 1)
    decay_xml.write_text(text)
    with pytest.raises(ValueError, match="Unknown entry inside PARAM"):
        read(decay_xml)


def test_check_name_uniqueness_rejects_collisions(decay_xml):
    # make an integrated variable collide with a trainable parameter name
    text = decay_xml.read_text().replace("<P> NAME = A </P>", "<P> NAME = k1 </P>", 1)
    decay_xml.write_text(text)
    reader = read(decay_xml)
    with pytest.raises(ValueError, match="not unique"):
        reader.check_name_uniqueness()


def test_check_name_uniqueness_passes_on_clean_input(decay_xml):
    read(decay_xml).check_name_uniqueness()  # must not raise


# ---------------------------------------------------------------------------
# optimizer settings — the previously untested fields
# ---------------------------------------------------------------------------

def test_population_settings(decay_xml):
    r = read(decay_xml)
    assert r.population_size == 20
    assert r.n_iters_pop == 5
    assert r.processors == 2
    assert r.pop_stepsize_rtol == [1e-5, 1e-5]
    assert r.pop_stepsize_atol == [1e-7, 1e-7]


def test_gradient_settings(decay_xml):
    r = read(decay_xml)
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


@pytest.mark.parametrize("value,expected", [("PSO", "PSO"), ("DE", "DE")])
def test_algorithm_is_parsed(decay_xml, value, expected):
    set_xml_setting(decay_xml, "POPULATION_OPT", "ALGORITHM", value)
    assert read(decay_xml).algorithm == expected


def test_random_seed_is_parsed(decay_xml):
    set_xml_setting(decay_xml, "POPULATION_OPT", "RANDOM_SEED", 7)
    assert read(decay_xml).random_seed == 7


def test_gradient_optimizer_is_lowercased(decay_xml):
    """The NODE class compares against lowercase names, so casing must not matter."""
    set_xml_setting(decay_xml, "GRADIENT_OPT", "GRADIENT_OPTIMIZER", "ADAM")
    assert read(decay_xml).gradient_optimizer == "adam"


def test_initial_time_is_parsed(decay_xml):
    set_xml_setting(decay_xml, "GRADIENT_OPT", "INITIAL_TIME", 0.5)
    assert read(decay_xml).init_time == 0.5


def test_unknown_population_key_raises(decay_xml):
    """POPULATION_OPT rejects unknown keys — this is what caught the stale
    NUM_PARTICLES key in the old test fixtures."""
    set_xml_setting(decay_xml, "POPULATION_OPT", "NUM_PARTICLES", 50)
    with pytest.raises(ValueError):
        read(decay_xml)


def test_unknown_gradient_key_is_silently_ignored(decay_xml):
    """Documented asymmetry: GRADIENT_OPT drops unknown keys instead of raising,
    so a typo'd gradient setting is silently not applied."""
    set_xml_setting(decay_xml, "GRADIENT_OPT", "MAXSTEPS", 999)
    r = read(decay_xml)
    assert r.max_steps == 4096  # the real MAX_STEPS, unaffected by the typo


def test_defaults_when_optional_settings_are_absent(decay_xml):
    for key in ("INTEGRATOR", "GRADIENT_OPTIMIZER"):
        remove_xml_setting(decay_xml, "GRADIENT_OPT", key)
    for key in ("POP_STEPSIZE_RTOL", "POP_STEPSIZE_ATOL"):
        remove_xml_setting(decay_xml, "POPULATION_OPT", key)
    r = read(decay_xml)
    assert r.integrator == "Kvaerno5"
    assert r.gradient_optimizer == "lbfgs"
    assert r.algorithm == "PSO"
    assert r.random_seed is None
    assert r.pop_stepsize_rtol is None
    assert r.pop_stepsize_atol is None


def test_write_results_flag(decay_xml):
    assert read(decay_xml).write_results is True
    decay_xml.write_text(decay_xml.read_text().replace(
        "<P> WRITE_RESULTS = Y </P>", "<P> WRITE_RESULTS = N </P>"))
    assert read(decay_xml).write_results is False


# ---------------------------------------------------------------------------
# experiments and initial conditions
# ---------------------------------------------------------------------------

def test_single_experiment(decay_xml):
    r = read(decay_xml)
    assert len(r.experiments) == 1
    assert r.experiments[0]["filename"] == "decay_data.csv"
    assert r.experiments[0]["ic_overrides"] == {}
    assert r.filename_data == "decay_data.csv"
    assert r.get_y0(0) == [1.0, 0.0]


def test_multi_experiment_initial_condition_overrides():
    """Per-experiment <INITIAL_CONDITIONS> must override only the named variables."""
    r = read(FIXTURES / "decay_multiexp" / "inputs" / "user_input.xml")
    assert len(r.experiments) == 2
    assert [e["filename"] for e in r.experiments] == ["decay_run_A.csv", "decay_run_B.csv"]

    assert r.experiments[0]["ic_overrides"] == {}
    assert r.get_y0(0) == [1.0, 0.0]          # falls back to the global INIT_VALs

    assert r.experiments[1]["ic_overrides"] == {"A": 2.0, "B": 0.5}
    assert r.get_y0(1) == [2.0, 0.5]          # overridden


def test_partial_initial_condition_override(tmp_path):
    """A variable absent from INITIAL_CONDITIONS keeps its global INIT_VAL."""
    import shutil

    xml = tmp_path / "user_input.xml"
    shutil.copy(FIXTURES / "decay_multiexp" / "inputs" / "user_input.xml", xml)
    text = xml.read_text().replace(
        """            <VAR>
                <P> NAME = B </P>
                <P> INIT_VAL = 0.5 </P>
            </VAR>
""", "")
    xml.write_text(text)

    r = read(xml)
    assert r.experiments[1]["ic_overrides"] == {"A": 2.0}
    assert r.get_y0(1) == [2.0, 0.0]  # A overridden, B from the global value


def test_unknown_initial_condition_name_raises(tmp_path):
    """A VAR name that matches no integrated variable must fail loudly in get_y0."""
    import shutil

    xml = tmp_path / "user_input.xml"
    shutil.copy(FIXTURES / "decay_multiexp" / "inputs" / "user_input.xml", xml)
    xml.write_text(xml.read_text().replace("<P> NAME = A </P>", "<P> NAME = typo </P>", 1))

    r = read(xml)
    with pytest.raises(ValueError):
        r.get_y0(1)


def test_get_y0_does_not_mutate_global_init_values():
    """get_y0 must return a copy; otherwise experiment 2's ICs would leak into
    experiment 1 on the next call."""
    r = read(FIXTURES / "decay_multiexp" / "inputs" / "user_input.xml")
    r.get_y0(1)
    assert r.integrated_variable_init_values == [1.0, 0.0]
    assert r.get_y0(0) == [1.0, 0.0]


def test_path_section_overrides_directory_names(tmp_path):
    xml = tmp_path / "p.xml"
    xml.write_text("""<?xml version="1.0" ?>
<FIT>
  <PATH>
    <P> USER_INPUT_DIR = in </P>
    <P> GENERATED_DIR = gen </P>
    <P> OUTPUT_DIR = out </P>
  </PATH>
</FIT>""")
    r = read(xml)
    assert (r.user_input_dirname, r.generated_dirname, r.output_dirname) == ("in", "gen", "out")
