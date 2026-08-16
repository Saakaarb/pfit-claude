# YAML input reader.
#
# This is the single point where user intent enters the framework: it parses
# sessions/<session>/inputs/user_input.yaml into the attribute surface that the
# rest of the pipeline consumes. Nothing downstream reads the file again.
#
# Two YAML-specific hazards drive the design, and both are handled here so no
# caller has to think about them:
#
#   1. PyYAML is YAML 1.1, which does NOT parse `1e-7`, `5E3` or `1e+7` as
#      floats — a float literal needs a dot in the mantissa AND a signed
#      exponent (`1.0e-7`). Every tolerance and bound in this project is written
#      in the failing form, so every numeric field is coerced explicitly through
#      _as_float / _as_int. Never trust the type PyYAML hands back.
#   2. Unquoted `on`, `off`, `yes`, `no`, `true`, `false` become booleans, as
#      both keys and values. A state variable or parameter named `on` would
#      silently arrive as `True`, so names are type-checked before use.
#
# Unknown keys raise in EVERY section. The XML reader this replaced was strict
# in most sections but silently dropped unknown keys in GRADIENT_OPT, so a
# typo'd MAX_STEPS vanished without warning; that asymmetry is deliberately gone.
from pathlib import Path

import yaml


class InputError(ValueError):
    """Raised for any malformed or incomplete user_input.yaml."""


# ---------------------------------------------------------------------------
# scalar coercion
# ---------------------------------------------------------------------------

def _as_float(value, where: str) -> float:
    """Coerce a YAML scalar to float, whatever type PyYAML decided it was.

    `1e-7` arrives as the string '1e-7' under YAML 1.1; float() accepts it.
    """
    if isinstance(value, bool):
        raise InputError(f"{where}: expected a number, got the boolean {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise InputError(f"{where}: expected a number, got {value!r}") from None


def _as_int(value, where: str) -> int:
    if isinstance(value, bool):
        raise InputError(f"{where}: expected an integer, got the boolean {value!r}")
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        raise InputError(f"{where}: expected an integer, got {value!r}") from None
    if as_float != int(as_float):
        raise InputError(f"{where}: expected an integer, got {value!r}")
    return int(as_float)


def _as_bool(value, where: str) -> bool:
    """Accept a real YAML bool, or the Y/N spelling carried over from the XML."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("y", "yes", "true", "on"):
            return True
        if text in ("n", "no", "false", "off"):
            return False
    raise InputError(
        f"{where}: expected true or false, got {value!r}"
    )


def _as_name(value, where: str) -> str:
    """A name that will become a dict key and an identifier in generated code.

    Rejects the YAML-bool coercions (`on`, `no`, ...) with an explanation,
    because the value the user typed and the value that arrives here differ.
    """
    if isinstance(value, bool):
        raise InputError(
            f"{where}: {'on/yes/true' if value else 'off/no/false'} is a YAML "
            f"boolean, not a name. Quote it (e.g. \"on\") if that is really the "
            f"intended name — but it must also be a valid Python identifier."
        )
    if not isinstance(value, str):
        raise InputError(f"{where}: expected a name, got {value!r}")
    name = value.strip()
    if not name.isidentifier():
        raise InputError(
            f"{where}: {name!r} is not a valid Python identifier. Names become "
            f"dict keys and identifiers in the generated script."
        )
    return name


# ---------------------------------------------------------------------------
# mapping helpers
# ---------------------------------------------------------------------------

def _as_mapping(value, where: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise InputError(f"{where}: expected a mapping, got {type(value).__name__}")
    for key in value:
        if not isinstance(key, str):
            raise InputError(
                f"{where}: key {key!r} is not a string. Unquoted on/off/yes/no "
                f"become booleans in YAML — quote such keys."
            )
    return value


def _as_list_of_mappings(value, where: str) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise InputError(
            f"{where}: expected a list of entries (order is load-bearing), got "
            f"{type(value).__name__}"
        )
    return [_as_mapping(item, f"{where}[{i}]") for i, item in enumerate(value)]


def _reject_unknown(mapping: dict, allowed: set, where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise InputError(
            f"{where}: unknown key(s) {unknown}. Valid keys are "
            f"{sorted(allowed)}."
        )


def _require(mapping: dict, key: str, where: str):
    if key not in mapping or mapping[key] is None:
        raise InputError(f"{where}: required key '{key}' is missing")
    return mapping[key]


def _tolerance_list(value, n_vars: int, where: str) -> list:
    """One tolerance per integrated variable; a scalar broadcasts to all of them."""
    if isinstance(value, (list, tuple)):
        values = [_as_float(v, f"{where}[{i}]") for i, v in enumerate(value)]
        if len(values) != n_vars:
            raise InputError(
                f"{where}: got {len(values)} value(s) but the model has {n_vars} "
                f"integrated variable(s). Give one value per variable, or a "
                f"single value to apply to all of them."
            )
        return values
    return [_as_float(value, where)] * n_vars


# ---------------------------------------------------------------------------
# the reader
# ---------------------------------------------------------------------------

TOP_LEVEL_SECTIONS = {
    "experiments", "model", "population_opt", "gradient_opt", "output", "paths",
}

EXPERIMENT_KEYS = {"data_file", "initial_conditions", "columns"}
COLUMN_KEYS = {"name", "units", "observes", "uncertainty_of"}
MODEL_KEYS = {"trainable_parameters", "fixed_parameters", "integrated_variables",
              "observables"}
OBSERVABLE_KEYS = {"name", "units"}
TRAINABLE_KEYS = {"name", "min_val", "max_val", "logscale"}
FIXED_KEYS = {"name", "value"}
VARIABLE_KEYS = {"name", "init_val"}
POPULATION_KEYS = {
    "population_size", "num_iters", "processors", "algorithm", "random_seed",
    "stepsize_rtol", "stepsize_atol",
}
GRADIENT_KEYS = {
    "num_iters", "stepsize_rtol", "stepsize_atol", "initial_timestep",
    "initial_time", "max_steps", "integrator", "gradient_optimizer",
    "init_value_lr", "end_value_lr", "transition_steps_lr", "decay_rate_lr",
}
OUTPUT_KEYS = {"write_results"}
PATHS_KEYS = {"user_input_dir", "generated_dir", "output_dir"}


class YAMLReader():

    def __init__(self):

        # path information
        self.user_input_dirname = "inputs"
        self.generated_dirname = "generated"
        self.output_dirname = "outputs"

        # model information
        self.n_search_axes = None
        self.min_axis_values = []
        self.max_axis_values = []
        self.axis_logscale = []
        self.trainable_parameter_names = []
        self.fixed_parameter_names = []
        self.fixed_parameter_values = []
        self.integrated_variable_names = []
        self.observable_names = []
        self.integrated_variable_init_values = []

        # population optimizer settings
        self.population_size = None
        self.n_iters_pop = None
        self.processors = None
        self.pop_stepsize_rtol = None  # if None, falls back to stepsize_rtol
        self.pop_stepsize_atol = None  # if None, falls back to stepsize_atol
        self.random_seed = None  # if set, makes the global search fully deterministic
        # gradient based optimizer settings
        self.n_iters_grad = None
        self.stepsize_rtol = None
        self.stepsize_atol = None
        self.init_timestep = None
        self.max_steps = None
        self.init_time = None
        self.init_value_lr = None
        self.end_value_lr = None
        self.transition_steps_lr = None
        self.decay_rate_lr = None
        self.integrator = "Kvaerno5"
        self.algorithm = "PSO"
        self.gradient_optimizer = 'lbfgs'
        self.error_loss = 5000.0

        # experiments: list of dicts, one per entry under `experiments`.
        # Each dict has the form:
        #   {
        #     'filename':      str,
        #     'ic_overrides':  dict[str, float]   # per-experiment IC overrides
        #   }
        # ic_overrides keys must match names in integrated_variable_names.
        # Any variable not listed in ic_overrides uses the global model value.
        self.experiments = []

        self.write_results = False
        self.output_dir = None

    @property
    def filename_data(self):
        """Backward-compatible accessor: returns the filename of the first experiment."""
        return self.experiments[0]['filename'] if self.experiments else None

    def get_y0(self, experiment_idx: int) -> list:
        """Return the initial condition vector for one experiment.

        Per-experiment ic_overrides take priority over the global values in
        model.integrated_variables.
        """
        y0 = list(self.integrated_variable_init_values)
        for var_name, val in self.experiments[experiment_idx]['ic_overrides'].items():
            idx = self.integrated_variable_names.index(var_name)
            y0[idx] = val
        return y0

    def read(self, config: dict) -> None:
        """Populate this reader from an already-loaded YAML mapping."""
        config = _as_mapping(config, "user_input.yaml")
        if not config:
            raise InputError("user_input.yaml is empty")
        _reject_unknown(config, TOP_LEVEL_SECTIONS, "user_input.yaml")

        # model must be read before the optimizer sections: the tolerance
        # lists are validated against the number of integrated variables.
        self._read_model(_as_mapping(_require(config, "model", "user_input.yaml"), "model"))
        self._read_experiments(config.get("experiments"))
        self._read_population_opt(
            _as_mapping(_require(config, "population_opt", "user_input.yaml"), "population_opt"))
        self._read_gradient_opt(
            _as_mapping(_require(config, "gradient_opt", "user_input.yaml"), "gradient_opt"))
        self._read_output(_as_mapping(config.get("output"), "output"))
        self._read_paths(_as_mapping(config.get("paths"), "paths"))

    # -- sections ----------------------------------------------------------

    def _read_model(self, model: dict) -> None:
        _reject_unknown(model, MODEL_KEYS, "model")

        trainable = _as_list_of_mappings(
            _require(model, "trainable_parameters", "model"), "model.trainable_parameters")
        if not trainable:
            raise InputError("model.trainable_parameters: at least one parameter is required")

        for i, param in enumerate(trainable):
            where = f"model.trainable_parameters[{i}]"
            _reject_unknown(param, TRAINABLE_KEYS, where)
            self.trainable_parameter_names.append(
                _as_name(_require(param, "name", where), f"{where}.name"))
            self.min_axis_values.append(
                _as_float(_require(param, "min_val", where), f"{where}.min_val"))
            self.max_axis_values.append(
                _as_float(_require(param, "max_val", where), f"{where}.max_val"))
            # stored as 0/1 rather than bool: it is consumed as a numeric flag
            # by jnp.where in the generated script's unscale_value.
            self.axis_logscale.append(
                int(_as_bool(_require(param, "logscale", where), f"{where}.logscale")))

        # Observables: quantities the model COMPUTES from the states and that
        # the data measures, e.g. an open probability or a relative percentage.
        # They are named here so a dataset column can reference one by name --
        # otherwise a derived column has only a free-text label and the link to
        # the model exists solely inside the loss body, where nothing can check
        # it. `_observables` in user_model.py must return exactly these keys.
        for i, observable in enumerate(_as_list_of_mappings(
                model.get("observables"), "model.observables")):
            where = f"model.observables[{i}]"
            _reject_unknown(observable, OBSERVABLE_KEYS, where)
            self.observable_names.append(
                _as_name(_require(observable, "name", where), f"{where}.name"))

        # The list length IS the parameter count. The XML carried a separate
        # N_TRAINABLE_PARAMETERS that had to agree with it; that redundancy
        # (and the mismatch error it caused) has no YAML equivalent.
        self.n_search_axes = len(trainable)

        for i, param in enumerate(_as_list_of_mappings(
                model.get("fixed_parameters"), "model.fixed_parameters")):
            where = f"model.fixed_parameters[{i}]"
            _reject_unknown(param, FIXED_KEYS, where)
            self.fixed_parameter_names.append(
                _as_name(_require(param, "name", where), f"{where}.name"))
            self.fixed_parameter_values.append(
                _as_float(_require(param, "value", where), f"{where}.value"))

        variables = _as_list_of_mappings(
            _require(model, "integrated_variables", "model"), "model.integrated_variables")
        if not variables:
            raise InputError("model.integrated_variables: at least one variable is required")

        for i, var in enumerate(variables):
            where = f"model.integrated_variables[{i}]"
            _reject_unknown(var, VARIABLE_KEYS, where)
            self.integrated_variable_names.append(
                _as_name(_require(var, "name", where), f"{where}.name"))
            self.integrated_variable_init_values.append(
                _as_float(_require(var, "init_val", where), f"{where}.init_val"))

    def _read_experiments(self, experiments) -> None:
        entries = _as_list_of_mappings(experiments, "experiments")
        if not entries:
            raise InputError("experiments: at least one experiment is required")

        for i, entry in enumerate(entries):
            where = f"experiments[{i}]"
            _reject_unknown(entry, EXPERIMENT_KEYS, where)

            filename = _require(entry, "data_file", where)
            if not isinstance(filename, str):
                raise InputError(f"{where}.data_file: expected a filename, got {filename!r}")

            ic_overrides = {}
            for name, value in _as_mapping(
                    entry.get("initial_conditions"), f"{where}.initial_conditions").items():
                var_name = _as_name(name, f"{where}.initial_conditions key")
                if var_name not in self.integrated_variable_names:
                    raise InputError(
                        f"{where}.initial_conditions: '{var_name}' is not an "
                        f"integrated variable. Known variables: "
                        f"{self.integrated_variable_names}"
                    )
                ic_overrides[var_name] = _as_float(
                    value, f"{where}.initial_conditions.{var_name}")

            self.experiments.append({
                'filename': filename.strip(),
                'ic_overrides': ic_overrides,
                'columns': self._read_columns(entry.get("columns"), where),
            })

    def _read_columns(self, columns, where: str) -> list:
        """Read one experiment's column declarations.

        Descriptive only: nothing here changes how the fit runs. The dataset's
        column meanings are otherwise recorded nowhere machine-readable -- the
        map from state to observable lives as arbitrary Python inside the loss
        -- so without this block, tooling cannot tell which column is which, and
        neither can a reader of the config.

        Entry i describes column i of the CSV, so the first entry is always the
        time column. `observes` names an integrated variable when the column is
        that state measured directly; a derived observable simply omits it.
        """
        entries = _as_list_of_mappings(columns, f"{where}.columns")
        if not entries:
            raise InputError(
                f"{where}.columns is required: declare one entry per column of "
                f"the CSV, starting with the time column. Without it the meaning "
                f"of each column is not recorded anywhere."
            )
        if len(entries) < 2:
            raise InputError(
                f"{where}.columns: expected at least 2 entries (time plus one "
                f"observable), got {len(entries)}"
            )

        declared = []
        for j, column in enumerate(entries):
            column_where = f"{where}.columns[{j}]"
            _reject_unknown(column, COLUMN_KEYS, column_where)
            name = _as_name(_require(column, "name", column_where), f"{column_where}.name")

            observes = column.get("observes")
            if observes is not None:
                observes = _as_name(observes, f"{column_where}.observes")
                known = self.integrated_variable_names + self.observable_names
                if observes not in known:
                    raise InputError(
                        f"{column_where}.observes: '{observes}' is neither an "
                        f"integrated variable nor a declared observable. Known: "
                        f"{known}"
                    )
                if j == 0:
                    raise InputError(
                        f"{column_where}.observes: column 0 is the time grid, "
                        f"not an observable, so it cannot observe a state"
                    )

            units = column.get("units")
            if units is not None and not isinstance(units, str):
                raise InputError(f"{column_where}.units: expected a string, got {units!r}")

            declared.append({
                'name': name,
                'units': units,
                'observes': observes,
                'uncertainty_of': column.get("uncertainty_of"),
            })

        # resolved in a second pass so a column may reference one declared after
        # it, and so the target is checked against the finished list
        measurements = {c['name'] for c in declared[1:] if c['uncertainty_of'] is None}
        for j, column in enumerate(declared):
            target = column['uncertainty_of']
            if target is None:
                continue
            column_where = f"{where}.columns[{j}]"
            target = _as_name(target, f"{column_where}.uncertainty_of")
            if j == 0:
                raise InputError(
                    f"{column_where}.uncertainty_of: column 0 is the time grid")
            if target == column['name']:
                raise InputError(
                    f"{column_where}.uncertainty_of: a column cannot be its own "
                    f"uncertainty")
            if target not in measurements:
                raise InputError(
                    f"{column_where}.uncertainty_of: '{target}' is not a measurement "
                    f"column of this experiment. Known measurement columns: "
                    f"{sorted(measurements)}"
                )
            column['uncertainty_of'] = target

        duplicates = {c['name'] for c in declared}
        if len(duplicates) != len(declared):
            raise InputError(f"{where}.columns: column names must be unique")
        return declared

    def _read_population_opt(self, settings: dict) -> None:
        _reject_unknown(settings, POPULATION_KEYS, "population_opt")
        n_vars = len(self.integrated_variable_names)

        self.population_size = _as_int(
            _require(settings, "population_size", "population_opt"),
            "population_opt.population_size")
        self.n_iters_pop = _as_int(
            _require(settings, "num_iters", "population_opt"), "population_opt.num_iters")
        self.processors = _as_int(
            _require(settings, "processors", "population_opt"), "population_opt.processors")

        if "algorithm" in settings:
            self.algorithm = str(settings["algorithm"]).strip()
        if settings.get("random_seed") is not None:
            self.random_seed = _as_int(settings["random_seed"], "population_opt.random_seed")
        if settings.get("stepsize_rtol") is not None:
            self.pop_stepsize_rtol = _tolerance_list(
                settings["stepsize_rtol"], n_vars, "population_opt.stepsize_rtol")
        if settings.get("stepsize_atol") is not None:
            self.pop_stepsize_atol = _tolerance_list(
                settings["stepsize_atol"], n_vars, "population_opt.stepsize_atol")

    def _read_gradient_opt(self, settings: dict) -> None:
        _reject_unknown(settings, GRADIENT_KEYS, "gradient_opt")
        n_vars = len(self.integrated_variable_names)

        self.n_iters_grad = _as_int(
            _require(settings, "num_iters", "gradient_opt"), "gradient_opt.num_iters")
        self.stepsize_rtol = _tolerance_list(
            _require(settings, "stepsize_rtol", "gradient_opt"), n_vars,
            "gradient_opt.stepsize_rtol")
        self.stepsize_atol = _tolerance_list(
            _require(settings, "stepsize_atol", "gradient_opt"), n_vars,
            "gradient_opt.stepsize_atol")
        self.init_timestep = _as_float(
            _require(settings, "initial_timestep", "gradient_opt"),
            "gradient_opt.initial_timestep")
        self.max_steps = _as_int(
            _require(settings, "max_steps", "gradient_opt"), "gradient_opt.max_steps")

        if settings.get("initial_time") is not None:
            self.init_time = _as_float(settings["initial_time"], "gradient_opt.initial_time")
        if "integrator" in settings:
            self.integrator = str(settings["integrator"]).strip()
        if "gradient_optimizer" in settings:
            self.gradient_optimizer = str(settings["gradient_optimizer"]).strip().lower()
        for key, attr in (("init_value_lr", "init_value_lr"),
                          ("end_value_lr", "end_value_lr"),
                          ("transition_steps_lr", "transition_steps_lr"),
                          ("decay_rate_lr", "decay_rate_lr")):
            if settings.get(key) is not None:
                setattr(self, attr, _as_float(settings[key], f"gradient_opt.{key}"))

    def _read_output(self, output: dict) -> None:
        _reject_unknown(output, OUTPUT_KEYS, "output")
        if "write_results" in output:
            self.write_results = _as_bool(output["write_results"], "output.write_results")

    def _read_paths(self, paths: dict) -> None:
        _reject_unknown(paths, PATHS_KEYS, "paths")
        if "user_input_dir" in paths:
            self.user_input_dirname = str(paths["user_input_dir"]).strip()
        if "generated_dir" in paths:
            self.generated_dirname = str(paths["generated_dir"]).strip()
        if "output_dir" in paths:
            self.output_dirname = str(paths["output_dir"]).strip()

    def check_name_uniqueness(self):
        combined_list = (self.trainable_parameter_names + self.fixed_parameter_names
                         + self.integrated_variable_names + self.observable_names)
        if len(combined_list) > len(set(combined_list)):
            raise ValueError("The provided names of every simulation element (trainable parameters, fixed parameters, integrated variables and observables) is not unique. Please provide unique names for each.")


def load_config(path) -> dict:
    """Read and parse a user_input.yaml, with the file named in any error."""
    path = Path(path)
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise InputError(f"{path}: not valid YAML: {exc}") from None


def read_input_file(path) -> YAMLReader:
    """Parse a user_input.yaml into a populated YAMLReader."""
    reader = YAMLReader()
    try:
        reader.read(load_config(path))
    except InputError as exc:
        raise InputError(f"{path}: {exc}") from None
    return reader
