import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from lib.utils.classes import ProblemObjectBase
from functools import partial
from lib.algorithms.PSO.classes import FitParamsPSO
from lib.algorithms.DE.classes import FitParamsDE
from lib.algorithms.NODE.classes import FitParamsNODE
from lib.utils.yamlread import YAMLReader, read_input_file
from pathlib import Path
import importlib.util
import sys

jax.config.update("jax_enable_x64", True)

class CreatedClass(ProblemObjectBase):
    def __init__(self, experiments: list, input_reader: YAMLReader, compute_loss_problem, write_problem_result):
        """
        Initialize the CreatedClass instance with problem configuration.

        Args:
            experiments (list[dict]): One dict per dataset file, each containing:
                't_eval'  (np.ndarray): time points for this experiment
                'dataset' (np.ndarray): data array of shape (time_steps, variables)
                'y0'      (jnp.ndarray): initial conditions for this experiment
            input_reader (YAMLReader): Configuration reader containing all problem parameters
            compute_loss_problem (callable): Function to compute loss for a single experiment
            write_problem_result (callable): Function to write results for a single experiment

        A separate constants dict is built for each experiment and stored in
        self.constants_list.  Loss is averaged over all experiments.
        """
        super().__init__()
        self.input_reader = input_reader
        self.experiments = experiments
        self.num_columns_to_fit = experiments[0]['dataset'].shape[1]
        self.params_to_fit_names = self.input_reader.trainable_parameter_names
        self.fixed_param_names = self.input_reader.fixed_parameter_names
        self.fixed_param_values = self.input_reader.fixed_parameter_values
        self.fixed_val_dict = {}
        for i in range(len(self.input_reader.fixed_parameter_names)):
            self.fixed_val_dict[self.fixed_param_names[i]] = self.fixed_param_values[i]

        self.constants_list = []
        for exp in experiments:
            c = {
                "dataset":          jnp.array(exp['dataset']),
                "t_eval":           jnp.array(exp['t_eval']),
                "init_cond":        exp['y0'],
                "num_steps":        exp['dataset'].shape[0],
                "init_time":        self.input_reader.init_time if self.input_reader.init_time is not None
                                    else exp['t_eval'][0],
                "final_time":       exp['t_eval'][-1],
                "stepsize_rtol":    np.array(self.input_reader.stepsize_rtol),
                "stepsize_atol":    np.array(self.input_reader.stepsize_atol),
                "init_timestep":    self.input_reader.init_timestep,
                "max_steps":        self.input_reader.max_steps,
                "fixed_parameters": self.fixed_val_dict,
                "error_loss":       self.input_reader.error_loss,
            }
            self.constants_list.append(c)

        self._compute_loss_problem = compute_loss_problem
        self._write_problem_result = write_problem_result

    def _compute_all_losses(self, population: np.ndarray)-> np.ndarray:
        """
        Compute losses for an entire population of parameter sets.

        Args:
            population (numpy.ndarray): Array of parameter sets, shape (n_individuals, n_parameters)

        Returns:
            numpy.ndarray: Array of loss values for each parameter set, with NaN/Inf values
                        replaced by 1e10 to prevent optimization issues

        This method iterates through each parameter set in the population and computes
        the corresponding loss using the JIT-compiled _compute_loss method.
        """
        #losses = []

        # get number of visible devices
        n_devices_total = len(jax.devices("cpu"))
        all_devices = jax.devices("cpu")

        user_selected_n_devices=int(self.input_reader.processors)

        if user_selected_n_devices > n_devices_total:
            print(f"User selected {user_selected_n_devices} devices, but only {n_devices_total} are available to JAX. Using all available devices.")

        used_devices = all_devices[:min(n_devices_total, user_selected_n_devices)]

        n_devices = len(used_devices)

        n_particles = population.shape[0]
        # define some array splitting logic
        local_n_particles = np.ceil(n_particles / n_devices).astype(int)
        pad = local_n_particles * n_devices - n_particles
        # pad to make equal-sized shards per device
        if pad:
            population = jnp.pad(population, ((0, pad), (0, 0)))

        # reshape to [n_dev, local_n, D] for pmap
        population_sharded = population.reshape(n_devices, local_n_particles, *population.shape[1:])

        # define a shard loss function
        #def shard_loss(shard_population: jax.Array)-> jax.Array:
            #losses = jnp.zeros(local_n_particles)
            #for i in range(shard_population.shape[0]):
            #    loss = self._compute_loss(shard_population[i])
                
                # JAX-compatible way to handle NaN/Inf values
            #    loss = jnp.where(jnp.logical_or(jnp.isnan(loss), jnp.isinf(loss)), 1e10, loss)
            #    losses = losses.at[i].set(loss)
        #    losses = jax.vmap(self._compute_loss)(shard_population)  # vectorized over the shard
        #    losses = jnp.where(jnp.isfinite(losses), losses, 1e10)     # sanitize NaN/Inf
        #    return losses

        # create a pmap with the number of devices
        # pmap over devices; broadcast data (in_axes=None)
        #print("Using devices: ",used_devices)
        #p_shard_loss = jax.pmap(shard_loss, in_axes=(0),devices=used_devices)
        #shard_losses = p_shard_loss(population_sharded)   

        # flatten back and drop padding
        #losses = shard_losses.reshape(-1)[:n_particles]

        if not hasattr(self, "_p_shard_loss") or getattr(self, "_p_shard_loss_axis", None) != n_devices:

            # Define once; use vmap to avoid Python loops/unrolling.
            def shard_loss(shard_population: jax.Array) -> jax.Array:
                # shard_population: [local_n_particles, d]
                losses = jax.vmap(self._compute_loss)(shard_population)  # vectorized over the shard
                losses = jnp.where(jnp.isfinite(losses), losses, 1e10)     # sanitize NaN/Inf
                return losses  # shape: [local_n_particles]

            # Create pmapped callable and cache it on self
            self._p_shard_loss = jax.pmap(shard_loss, in_axes=0, devices=used_devices)
            self._p_shard_loss_axis = n_devices

        # --- run pmapped compute ---
        shard_losses = self._p_shard_loss(population_sharded)  # [n_devices, local_n_particles]
        losses = shard_losses.reshape(-1)[:n_particles]
        
        return np.array(losses)

    @partial(jax.jit, static_argnums=(0,))
    def _compute_loss(self, design_pt: np.ndarray)-> float:
        """
        Compute loss for a single parameter set using JIT compilation.

        Loss is averaged over all experiments in self.constants_list.
        The Python loop is unrolled at trace time, producing a single fused
        computation graph across all experiments.

        Args:
            design_pt (jax.numpy.ndarray): Single parameter set to evaluate

        Returns:
            float: Mean loss over all experiments
        """
        losses = [self._compute_loss_problem(c, jnp.array(design_pt))
                  for c in self.constants_list]
        return jnp.mean(jnp.array(losses))

    def set_min_limit(self, min_lim: list[float])-> None:
        """
        Set minimum bounds for parameter search space across all experiments.

        Args:
            min_lim (numpy.ndarray): Array of minimum values for each parameter
        """
        for c in self.constants_list:
            c["min_limits"] = jnp.array(min_lim)

    def set_max_limit(self, max_lim: list[float])-> None:
        """
        Set maximum bounds for parameter search space across all experiments.

        Args:
            max_lim (numpy.ndarray): Array of maximum values for each parameter
        """
        for c in self.constants_list:
            c["max_limits"] = jnp.array(max_lim)

    def set_is_logscale(self, is_logscale: list[bool])-> None:
        """
        Set log-scale flag for parameter axes across all experiments.

        Args:
            is_logscale (numpy.ndarray): Boolean array indicating which parameters
                                        should use log-scale transformation
        """
        for c in self.constants_list:
            c["is_logscale"] = jnp.array(is_logscale)

    def write_problem_result(self, design_point: np.ndarray, input_reader: YAMLReader, label:str="default")-> None:
        """
        Write problem solution results to CSV files, one file per experiment.

        Args:
            design_point (numpy.ndarray): Parameter set that produced the solution
            input_reader (YAMLReader): Configuration reader containing output directory info
            label (str, optional): Label prefix for the output files. Defaults to "default"

        Output files are named "{label}_solution_exp1.csv", "_exp2.csv", etc.
        For a single experiment the file is named "{label}_solution_exp1.csv".
        """
        for i, c in enumerate(self.constants_list):
            writeout_array = self._write_problem_result(c, jnp.array(design_point))
            np.savetxt(input_reader.output_dir/Path(f"{label}_solution_exp{i+1}.csv"),
                       writeout_array, delimiter=",")
      


def get_input_reader(path_to_input: Path)-> YAMLReader:
    """
    Parse the YAML input file and create a YAMLReader instance.

    Args:
        path_to_input (Path): Path to the YAML configuration file

    Returns:
        YAMLReader: Configured reader instance containing all problem parameters

    Raises:
        InputError: if the file is not valid YAML, or a required key is missing,
        or an unknown key is present in any section.
    """
    return read_input_file(path_to_input)


def fit_generic_system(path_to_input: Path, path_to_output_dir: Path, generated_dir: Path,session_path: Path)-> np.ndarray:
    """Fit a generic system using a two-phase optimization approach.

    This function performs parameter fitting using a combination of:
    1. Population-based optimization (PSO) for global search
    2. Gradient-based optimization (NODE) for local refinement

    The process includes:
    1. Reading input parameters from user_input.yaml
    2. Running PSO to find initial parameter estimates
    3. Using PSO results as initial guess for NODE
    4. Running NODE to refine the parameters
    5. Writing results to output directory

    Parameters
    ----------
    path_to_input : str or Path
        Path to the input YAML file containing optimization parameters
    path_to_output_dir : str or Path
        Directory where output files will be written
    generated_dir : str or Path
        Directory containing generated files (user_model.py, etc.)

    Notes
    -----
    - PSO is used first to explore the parameter space globally
    - NODE uses the best PSO result as its initial guess
    - Results are written to the output directory:
        - final_design_point.csv : Best parameters found
        - result_solution.csv : Solution trajectory
        - fitting_error.txt : Error messages if any
    - Progress is logged to the output directory

    Returns
    -------
    numpy.ndarray: Best parameter set found during optimization

    See Also
    --------
    FitParamsPSO : Class for PSO optimization parameters
    FitParamsNODE : Class for NODE optimization parameters
    """
    try:
        # Dynamically import generated_script.py from the session's generated_dir
        generated_script_path = Path(generated_dir) / "generated_script.py"
        spec = importlib.util.spec_from_file_location("generated_script", generated_script_path)
        generated_script = importlib.util.module_from_spec(spec)
        sys.modules["generated_script"] = generated_script
        spec.loader.exec_module(generated_script)

        input_reader=get_input_reader(path_to_input)

        # variable and parameter name uniqueness
        input_reader.check_name_uniqueness()

        # assign output dir
        input_reader.output_dir=path_to_output_dir

        # load all experiment datasets
        experiments_data = []
        for i, exp in enumerate(input_reader.experiments):
            dataset_path = session_path / Path(input_reader.user_input_dirname) / Path(exp['filename'])
            with open(dataset_path, 'r', encoding='utf-8-sig') as f:
                all_data = np.genfromtxt(f, dtype=float, delimiter=',')
            experiments_data.append({
                't_eval':  all_data[:, 0],
                'dataset': all_data[:, 1:],
                'y0':      jnp.array(input_reader.get_y0(i)),
            })

        problem_obj = CreatedClass(experiments=experiments_data, input_reader=input_reader,
                                   compute_loss_problem=generated_script._compute_loss_problem,
                                   write_problem_result=generated_script._write_problem_result)

        final_ans = fit_equation_system(input_reader, problem_obj)

        return final_ans
        
    except Exception as e:
        error_message = f"Fitting process failed with error: {str(e)}"
        print(error_message)
        
        # Write error to file
        error_file = Path(path_to_output_dir) / "fitting_error.txt"
        with open(error_file, 'w') as f:
            f.write(error_message)
        
        # Re-raise the exception to ensure the process exits with error code
        raise e


# fixed
def fit_equation_system(input_reader: YAMLReader, problem_obj: CreatedClass)-> np.ndarray:
    """
    Fit a system of equations using a two-phase optimization approach.

    This function performs parameter fitting for a system of equations using:
    1. Population-based optimization (PSO) for global search
    2. Gradient-based optimization (NODE) for local refinement

    The process includes:
    1. Running PSO to find initial parameter estimates
    2. Using PSO results as initial guess for NODE
    3. Running NODE to refine the parameters
    4. Writing results to output directory

    Args:
        input_reader (YAMLReader): Reader object containing optimization parameters from user_input.yaml
        problem_obj (CreatedClass): Problem object built from all experiment datasets

    Returns:
        numpy.ndarray: Best parameter set found during optimization

    Notes:
        - PSO is used first to explore the parameter space globally
        - NODE uses the best PSO result as its initial guess
        - Results are written to the output directory specified in input_reader
        - Progress is logged to the output directory
        - Final parameters are saved to final_design_point.csv
    """
    algorithm = getattr(input_reader, 'algorithm', 'PSO').upper()

    if algorithm == 'DE':
        fit_obj = FitParamsDE(input_reader, problem_obj)
    else:
        fit_obj = FitParamsPSO(input_reader, problem_obj)

    # use loose tolerances for global search if specified, else fall back to gradient tolerances
    pop_rtol = input_reader.pop_stepsize_rtol or input_reader.stepsize_rtol
    pop_atol = input_reader.pop_stepsize_atol or input_reader.stepsize_atol
    for c in problem_obj.constants_list:
        c['stepsize_rtol'] = jnp.array(pop_rtol)
        c['stepsize_atol'] = jnp.array(pop_atol)
    problem_obj.set_min_limit(fit_obj.min_search_axis)
    problem_obj.set_max_limit(fit_obj.max_search_axis)
    problem_obj.set_is_logscale(input_reader.axis_logscale)

    print(f"Global search tolerances — rtol: {pop_rtol}, atol: {pop_atol}")

    if algorithm == 'DE':
        print("Writing de log file")
        log_path = Path(input_reader.output_dir) / "de_fitting.log"
        with open(log_path, 'w') as log_file:
            log_file.write(f"Total DE iterations: {input_reader.n_iters_pop}\n")
            log_file.write("-" * 50 + "\n\n")
        best_position, best_cost = fit_obj.run(log_path)
    else:
        print("Writing pso log file")
        log_path = Path(input_reader.output_dir) / "pso_fitting.log"
        with open(log_path, 'w') as log_file:
            log_file.write(f"Total number of PSO iterations: {input_reader.n_iters_pop}\n")
            log_file.write("-" * 50 + "\n\n")
        best_position, best_cost = optimize_function(fit_obj, input_reader, log_path)

    unscaled_best_position = fit_obj.unscale_design_point(best_position)
    print(f"Best Position from {algorithm}:", unscaled_best_position)
    print(f"Best cost from {algorithm}:", best_cost)

    # NODE uses a separate problem object so its JIT compilation bakes in tight tolerances
    problem_obj_node = CreatedClass(
        experiments=problem_obj.experiments,
        input_reader=input_reader,
        compute_loss_problem=problem_obj._compute_loss_problem,
        write_problem_result=problem_obj._write_problem_result,
    )
    problem_obj_node.set_min_limit(fit_obj.min_search_axis)
    problem_obj_node.set_max_limit(fit_obj.max_search_axis)
    problem_obj_node.set_is_logscale(input_reader.axis_logscale)
    print(f"NODE tolerances — rtol: {input_reader.stepsize_rtol}, atol: {input_reader.stepsize_atol}")

    fit_obj_NODE = FitParamsNODE(
        input_reader, problem_obj_node, init_guess=unscaled_best_position
    )

    try:
        tuned_best_position, tuned_best_loss = fit_obj_NODE.train_NODE()
    except Exception as e:
        print(f"Error in NODE training: {e}, stopping")
        tuned_best_position = best_position
        tuned_best_loss = 1e10

    print("Tuned position from NODE(scaled):", tuned_best_position)
    print("Tuned best loss:", tuned_best_loss)

    unscaled_best_position_tuned = fit_obj.unscale_design_point(
        np.array(tuned_best_position)
    )
    print("Final best position:", unscaled_best_position_tuned)

    if input_reader.write_results:
        problem_obj_node.write_problem_result(tuned_best_position, input_reader, label="result")

    # save design point to file
    unscaled_best_position_tuned_np=np.array(unscaled_best_position_tuned)
    #TODO edit this to save the number obtained with the name of each parameter
    np.savetxt(input_reader.output_dir/Path(f"final_design_point.csv"),unscaled_best_position_tuned_np,delimiter=",")

    # post-fit sloppiness / identifiability diagnostic (never fatal)
    from lib.utils.sloppiness import run_sloppiness_analysis
    run_sloppiness_analysis(problem_obj_node._compute_loss_problem, problem_obj_node.constants_list,
                            np.array(tuned_best_position), input_reader.trainable_parameter_names,
                            input_reader.output_dir)

    return unscaled_best_position_tuned


def fit_gradient_only_system(path_to_input: Path, path_to_output_dir: Path, generated_dir: Path,
                             session_path: Path, init_guess: np.ndarray)-> np.ndarray:
    """Run ONLY the gradient (NODE) refinement stage, seeded from an existing point.

    This skips the population-based global search entirely and starts the
    gradient optimizer directly from ``init_guess`` (in real parameter units).
    It is the entry point used by ``fit_gradient_only.py`` so a user can cheaply
    re-refine a previously found design point (e.g. with a different optimizer,
    more iterations, or tighter tolerances) without repeating the expensive
    PSO/DE search.

    Unlike ``fit_generic_system`` this does NOT wipe the output directory — the
    population-search logs and the ``final_design_point.csv`` that seeds this run
    are preserved. Only the NODE outputs (``NODE_fitting.log``,
    ``result_solution_expN.csv``, ``final_design_point.csv``) are overwritten.

    Parameters
    ----------
    path_to_input : Path
        Path to the session's user_input.yaml.
    path_to_output_dir : Path
        Session output directory (must already exist).
    generated_dir : Path
        Directory containing generated_script.py.
    session_path : Path
        Session root, used to resolve experiment data files.
    init_guess : numpy.ndarray
        Starting parameter set in real (unscaled) units, in YAML trainable order.

    Returns
    -------
    numpy.ndarray
        The refined parameter set in real units.
    """
    try:
        generated_script_path = Path(generated_dir) / "generated_script.py"
        spec = importlib.util.spec_from_file_location("generated_script", generated_script_path)
        generated_script = importlib.util.module_from_spec(spec)
        sys.modules["generated_script"] = generated_script
        spec.loader.exec_module(generated_script)

        input_reader = get_input_reader(path_to_input)
        input_reader.check_name_uniqueness()
        input_reader.output_dir = path_to_output_dir

        # load all experiment datasets (identical to fit_generic_system)
        experiments_data = []
        for i, exp in enumerate(input_reader.experiments):
            dataset_path = session_path / Path(input_reader.user_input_dirname) / Path(exp['filename'])
            with open(dataset_path, 'r', encoding='utf-8-sig') as f:
                all_data = np.genfromtxt(f, dtype=float, delimiter=',')
            experiments_data.append({
                't_eval':  all_data[:, 0],
                'dataset': all_data[:, 1:],
                'y0':      jnp.array(input_reader.get_y0(i)),
            })

        # NODE problem object is built with the (tight) gradient tolerances that
        # CreatedClass reads from input_reader.stepsize_rtol/atol by default.
        problem_obj_node = CreatedClass(
            experiments=experiments_data, input_reader=input_reader,
            compute_loss_problem=generated_script._compute_loss_problem,
            write_problem_result=generated_script._write_problem_result)

        init_guess = np.asarray(init_guess, dtype=float)
        if init_guess.shape[0] != input_reader.n_search_axes:
            raise ValueError(
                f"init_guess has {init_guess.shape[0]} entries but user_input.yaml defines "
                f"{input_reader.n_search_axes} trainable parameters")

        fit_obj_NODE = FitParamsNODE(input_reader, problem_obj_node, init_guess=init_guess)
        problem_obj_node.set_min_limit(fit_obj_NODE.min_search_axis)
        problem_obj_node.set_max_limit(fit_obj_NODE.max_search_axis)
        problem_obj_node.set_is_logscale(input_reader.axis_logscale)

        print(f"Gradient-only run — optimizer: {input_reader.gradient_optimizer}, "
              f"iters: {input_reader.n_iters_grad}")
        print(f"NODE tolerances — rtol: {input_reader.stepsize_rtol}, atol: {input_reader.stepsize_atol}")
        print(f"Initial guess (real units): {init_guess}")

        tuned_best_position, tuned_best_loss = fit_obj_NODE.train_NODE()

        print("Tuned position from NODE(scaled):", tuned_best_position)
        print("Tuned best loss:", tuned_best_loss)

        unscaled_best_position_tuned = fit_obj_NODE.unscale_design_point(
            np.array(tuned_best_position))
        print("Final best position:", unscaled_best_position_tuned)

        if input_reader.write_results:
            problem_obj_node.write_problem_result(tuned_best_position, input_reader, label="result")

        unscaled_best_position_tuned_np = np.array(unscaled_best_position_tuned)
        np.savetxt(input_reader.output_dir / Path("final_design_point.csv"),
                   unscaled_best_position_tuned_np, delimiter=",")

        # post-fit sloppiness / identifiability diagnostic (never fatal)
        from lib.utils.sloppiness import run_sloppiness_analysis
        run_sloppiness_analysis(problem_obj_node._compute_loss_problem, problem_obj_node.constants_list,
                                np.array(tuned_best_position), input_reader.trainable_parameter_names,
                                input_reader.output_dir)

        return unscaled_best_position_tuned

    except Exception as e:
        error_message = f"Gradient-only fitting process failed with error: {str(e)}"
        print(error_message)
        error_file = Path(path_to_output_dir) / "fitting_error.txt"
        with open(error_file, 'w') as f:
            f.write(error_message)
        raise e


def optimize_function(fit_obj: FitParamsPSO, input_reader: YAMLReader, file_obj: Path)-> tuple[np.ndarray, float]:
    """
    Execute PSO optimization iterations with logging and error handling.

    This function runs the PSO optimization algorithm for the specified number
    of iterations, logging progress and handling any errors that occur during
    the optimization process.

    Args:
        fit_obj (FitParamsPSO): PSO optimization object configured with problem parameters
        input_reader (YAMLReader): Configuration reader containing iteration count and output directory
        file_obj (Path): Path to the log file for writing optimization progress

    Returns:
        tuple: (best_position, best_cost) - Best parameter set found and its corresponding cost

    Raises:
        Exception: If optimization fails, with error details written to fitting_error.txt

    Notes:
        - The function checks for a stop_fitting.flag file to allow early termination
        - Progress is logged to the specified log file
        - Errors are captured and written to fitting_error.txt in the output directory
        - If optimization fails, the current best position is returned if available
    """  
    print("Starting Search Iterations")
    # Iterations
    try:
        for iter_no in range(input_reader.n_iters_pop):
            
            # Check for stop flag
            stop_flag = Path(input_reader.output_dir) / "stop_fitting.flag"
            if stop_flag.exists():
                print("Stop flag detected, stopping PSO optimization")
                break

            fit_obj.search_iteration(iter_no, file_obj)
    except Exception as e:
        error_message = f"PSO optimization failed at iteration {iter_no}: {str(e)}"
        print(error_message)
        
        # Write error to file
        error_file = Path(input_reader.output_dir) / "fitting_error.txt"
        with open(error_file, 'w') as f:
            f.write(error_message)
        
        # Return current best position if available, otherwise raise
        if hasattr(fit_obj, 'best_pos') and fit_obj.best_pos is not None:
            return fit_obj.best_pos, fit_obj.swarm_obj.best_cost
        else:
            raise e

    return fit_obj.best_pos,fit_obj.swarm_obj.best_cost


# class CreatedClass(ProblemObjectBase):
#     def __init__(self, dataset, t_eval, y0, input_reader, compute_loss_problem, write_problem_result):
#         super().__init__()
#         self.y0 = y0
#         self.input_reader = input_reader
#         self.t_eval = t_eval
#         self.dataset = np.array(dataset)
#         self.num_columns_to_fit = self.dataset.shape[1]
#         self.params_to_fit_names = self.input_reader.trainable_parameter_names
#         self.fixed_param_names = self.input_reader.fixed_parameter_names
#         self.fixed_param_values = self.input_reader.fixed_parameter_values
#         self.fixed_val_dict = {}
#         for i in range(len(self.input_reader.fixed_parameter_names)):
#             self.fixed_val_dict[self.fixed_param_names[i]] = self.fixed_param_values[i]
#         self.constants = {
#             "dataset": jnp.array(self.dataset),
#             "t_eval": t_eval,
#             "init_cond": y0,
#         }
#         print("NOTE: currently, only simulations till the same final time are supported")
#         self.constants["num_steps"] = self.dataset.shape[0]
#         if self.input_reader.init_time is None:
#             self.constants["init_time"] = self.t_eval[0]
#         else:
#             self.constants["init_time"] = self.input_reader.init_time
#         self.constants["final_time"] = self.t_eval[-1]
#         self.constants['stepsize_rtol'] = np.array(self.input_reader.stepsize_rtol)
#         self.constants['stepsize_atol'] = np.array(self.input_reader.stepsize_atol)
#         self.constants['init_timestep'] = self.input_reader.init_timestep
#         self.constants['max_steps'] = self.input_reader.max_steps
#         self.constants['fixed_parameters'] = self.fixed_val_dict
#         self.constants['error_loss'] = self.input_reader.error_loss
#         self._compute_loss_problem = compute_loss_problem
#         self._write_problem_result = write_problem_result

#     def _compute_all_losses(self, population):
#         losses = []
#         for i in range(population.shape[0]):
#             loss = self._compute_loss(population[i])
            
#             if np.isnan(loss) or np.isinf(loss):
#                 loss=1e10
#             losses.append(loss)
#         return np.array(losses)

#     @partial(jax.jit, static_argnums=(0,))
#     def _compute_loss(self, design_pt):
#         return self._compute_loss_problem(self.constants, jnp.array(design_pt))

#     def set_min_limit(self, min_lim):
#         self.constants["min_limits"] = jnp.array(min_lim)

#     def set_max_limit(self, max_lim):
#         self.constants["max_limits"] = jnp.array(max_lim)

#     def set_is_logscale(self, is_logscale):
#         self.constants["is_logscale"] = jnp.array(is_logscale)

#     def write_problem_result(self, design_point, input_reader, label="default"):
#         writeout_array = self._write_problem_result(self.constants, jnp.array(design_point))
#         np.savetxt(input_reader.output_dir/Path(f"{label}_solution.csv"), writeout_array, delimiter=",")
        
        