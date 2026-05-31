# XML Reader file
from copy import deepcopy
import re

class XMLReader():

    def __init__(self):

        # path information
        self.user_input_dirname="inputs"
        self.generated_dirname="generated"
        self.output_dirname="outputs"

        # model information
        self.n_search_axes=None
        self.min_axis_values=[]
        self.max_axis_values=[]
        self.axis_logscale=[]
        self.trainable_parameter_names=[]
        self.fixed_parameter_names=[]
        self.fixed_parameter_values=[]
        self.integrated_variable_names=[]
        self.integrated_variable_init_values=[]

        # TODO read a boundary condition for DAEs

        # population optimizer settings
        self.n_particles=None
        self.n_iters_pop=None
        self.processors=None
        self.pso_stepsize_rtol=None  # if None, falls back to stepsize_rtol
        self.pso_stepsize_atol=None  # if None, falls back to stepsize_atol
        # gradient based optimizer settings
        self.n_iters_grad=None
        self.stepsize_rtol=None
        self.stepsize_atol=None
        self.init_timestep=None
        self.max_steps=None
        self.init_time=None
        self.init_value_lr=None
        self.end_value_lr=None
        self.transition_steps_lr=None
        self.decay_rate_lr=None
        self.integrator="Kvaerno5"
        self.error_loss=5000.0

        # experiments: list of dicts, one per <EXPERIMENT> block in the XML.
        # Each dict has the form:
        #   {
        #     'filename':        str,
        #     'column_names':    list[str],
        #     'column_indices':  list[int],
        #     'ic_overrides':    dict[str, float]   # per-experiment IC overrides
        #   }
        # ic_overrides keys must match names in integrated_variable_names.
        # Any variable not listed in ic_overrides uses the global INTEGRATED_SYSTEM_DESCRIPTION value.
        self.experiments = []

        self.write_results = False
        self.output_dir = None

    @property
    def filename_data(self):
        """Backward-compatible accessor: returns the filename of the first experiment."""
        return self.experiments[0]['filename'] if self.experiments else None

    def get_y0(self, experiment_idx: int) -> list:
        """Return the initial condition vector for one experiment.

        Per-experiment ic_overrides (from <INITIAL_CONDITIONS>) take priority
        over the global values in <INTEGRATED_SYSTEM_DESCRIPTION>.
        """
        y0 = list(self.integrated_variable_init_values)
        for var_name, val in self.experiments[experiment_idx]['ic_overrides'].items():
            idx = self.integrated_variable_names.index(var_name)
            y0[idx] = val
        return y0

    def get_trailing_number(self, s):
        m = re.search(r'\d+$', s)
        return int(m.group()) if m else None

    def read_XML(self, root):

        for child in root:

            if child.tag == "EXPERIMENT":
                exp = {
                    'filename': None,
                    'column_names': [],
                    'column_indices': [],
                    'ic_overrides': {}
                }
                for child2 in child:

                    if child2.tag == "FILENAME":
                        for child3 in child2:
                            name = child3.text.split('=')[0].strip()
                            value = child3.text.split('=')[1].strip()
                            if name == "FILENAME_DATA":
                                exp['filename'] = value

                    elif child2.tag == "COLUMN_INFO":
                        for child3 in child2:
                            for child4 in child3:
                                name = child4.text.split('=')[0].strip()
                                value = child4.text.split('=')[1].strip()
                            if name == "NAME":
                                exp['column_names'].append(value)
                            elif name == "COLUMN_INDEX":
                                exp['column_indices'].append(int(value))

                    elif child2.tag == "INITIAL_CONDITIONS":
                        for child3 in child2:
                            ic_name = None
                            ic_val = None
                            for child4 in child3:
                                k = child4.text.split('=')[0].strip()
                                v = child4.text.split('=')[1].strip()
                                if k == "NAME":
                                    ic_name = v
                                elif k == "INIT_VAL":
                                    ic_val = float(v)
                            if ic_name is not None and ic_val is not None:
                                exp['ic_overrides'][ic_name] = ic_val

                self.experiments.append(exp)

            elif child.tag == "PATH":
                for child2 in child:
                    name = child2.text.split('=')[0].strip()
                    value = child2.text.split('=')[1].strip()
                    if name == "USER_INPUT_DIR":
                        self.user_input_dirname = value
                    elif name == "GENERATED_DIR":
                        self.generated_dirname = value
                    elif name == "OUTPUT_DIR":
                        self.output_dirname = value

            elif child.tag == "MODEL":

                for child2 in child:

                    if child2.tag == "TRAINABLE_PARAMETER_DESCRIPTION":

                        for child3 in child2:

                            for child4 in child3:
                                name = child4.text.split('=')[0].strip()
                                value = child4.text.split('=')[1].strip()
                                if name == "PARAMETER_NAME":
                                    self.trainable_parameter_names.append(value)
                                elif name == "MIN_VAL":
                                    self.min_axis_values.append(float(value))
                                elif name == "MAX_VAL":
                                    self.max_axis_values.append(float(value))
                                elif name == "LOGSCALE":
                                    if value == "N":
                                        self.axis_logscale.append(int(0))
                                    elif value == "Y":
                                        self.axis_logscale.append(int(1))
                                    else:
                                        raise ValueError("Unknown logscale value")
                                else:
                                    raise ValueError("Unknown entry inside PARAM")

                    elif child2.tag == "INTEGRATED_SYSTEM_DESCRIPTION":

                        for child3 in child2:

                            for child4 in child3:
                                name = child4.text.split('=')[0].strip()
                                value = child4.text.split('=')[1].strip()
                                if name == "NAME":
                                    self.integrated_variable_names.append(value)
                                elif name == "INIT_VAL":
                                    self.integrated_variable_init_values.append(float(value))

                    elif child2.tag == "TRAINABLE_PARAMETERS":

                        for child3 in child2:
                            name = child3.text.split('=')[0].strip()
                            value = child3.text.split('=')[1].strip()
                            if name == "N_TRAINABLE_PARAMETERS":
                                self.n_search_axes = int(value)
                            else:
                                raise ValueError

                    elif child2.tag == "FIXED_PARAM_DESCRIPTION":

                        for child3 in child2:
                            for child4 in child3:
                                name = child4.text.split('=')[0].strip()
                                value = child4.text.split('=')[1].strip()
                                if name == "NAME":
                                    self.fixed_parameter_names.append(value)
                                elif name == "VALUE":
                                    self.fixed_parameter_values.append(float(value))

            elif child.tag == "POPULATION_OPT":

                for child2 in child:

                    if child2.tag == "SETTINGS":

                        for child3 in child2:
                            name = child3.text.split('=')[0].strip()
                            value = child3.text.split('=')[1].strip()

                            if name == "NUM_PARTICLES":
                                self.n_particles = int(value)
                            elif name == "NUM_ITERS":
                                self.n_iters_pop = int(value)
                            elif name == "PROCESSORS":
                                self.processors = int(value)
                            elif name == "PSO_STEPSIZE_RTOL":
                                self.pso_stepsize_rtol = [float(x.strip()) for x in value.split(',')]
                            elif name == "PSO_STEPSIZE_ATOL":
                                self.pso_stepsize_atol = [float(x.strip()) for x in value.split(',')]
                            else:
                                raise ValueError

            elif child.tag == "GRADIENT_OPT":

                for child2 in child:

                    if child2.tag == "SETTINGS":

                        for child3 in child2:
                            name = child3.text.split('=')[0].strip()
                            value = child3.text.split('=')[1].strip()

                            if name == "NUM_ITERS":
                                self.n_iters_grad = int(value)
                            elif name == "STEPSIZE_RTOL":
                                self.stepsize_rtol = [float(x.strip()) for x in value.split(',')]
                            elif name == "STEPSIZE_ATOL":
                                self.stepsize_atol = [float(x.strip()) for x in value.split(',')]
                            elif name == "INITIAL_TIMESTEP":
                                self.init_timestep = float(value)
                            elif name == "INITIAL_TIME":
                                self.init_time = float(value)
                            elif name == "MAX_STEPS":
                                self.max_steps = int(value)
                            elif name == "INIT_VALUE_LR":
                                self.init_value_lr = float(value)
                            elif name == "END_VALUE_LR":
                                self.end_value_lr = float(value)
                            elif name == "TRANSITION_STEPS_LR":
                                self.transition_steps_lr = float(value)
                            elif name == "DECAY_RATE_LR":
                                self.decay_rate_lr = float(value)
                            elif name == "INTEGRATOR":
                                self.integrator = value

            elif child.tag == "PLOTTING_INFO":

                for child2 in child:
                    name = child2.text.split('=')[0].strip()
                    value = child2.text.split('=')[1].strip()
                    if name == "WRITE_RESULTS":
                        if value == "Y":
                            self.write_results = True
                        elif value == "N":
                            self.write_results = False

    def check_name_uniqueness(self):
        combined_list = self.trainable_parameter_names + self.fixed_parameter_names + self.integrated_variable_names
        if len(combined_list) > len(set(combined_list)):
            raise ValueError("The provided names of every simulation element (trainable parameters, fixed parameters and integrated variable name) is not unique. Please provide unique names for each.")
