import numpy as np
import scipy.optimize
import time
from lib.utils.yamlread import YAMLReader
from lib.utils.classes import ProblemObjectBase
from lib.utils.doe_space_sampling import get_lhs_sampling
from pathlib import Path


class FitParamsDE:

    def __init__(self, input_reader: YAMLReader, problem_object: ProblemObjectBase):
        self.problem_obj = problem_object
        self.input_reader = input_reader
        # Fall back to 42 when unset so DE stays reproducible by default (its
        # historical behavior); an explicit RANDOM_SEED overrides it.
        self.random_seed = getattr(input_reader, "random_seed", None)
        if self.random_seed is None:
            self.random_seed = 42

        self.min_search_list = []
        self.max_search_list = []
        self.min_search_axis = []
        self.max_search_axis = []
        self.axis_logscale = input_reader.axis_logscale

        for i_axis in range(input_reader.n_search_axes):
            if input_reader.axis_logscale[i_axis]:
                curr_axis_min = np.log10(input_reader.min_axis_values[i_axis])
                curr_axis_max = np.log10(input_reader.max_axis_values[i_axis])
            else:
                curr_axis_min = input_reader.min_axis_values[i_axis]
                curr_axis_max = input_reader.max_axis_values[i_axis]

            self.min_search_axis.append(curr_axis_min)
            self.max_search_axis.append(curr_axis_max)

            sc_min = self.scale_value(curr_axis_min, curr_axis_min, curr_axis_max)
            sc_max = self.scale_value(curr_axis_max, curr_axis_min, curr_axis_max)
            self.min_search_list.append(sc_min)
            self.max_search_list.append(sc_max)

        self.n_search_axes = input_reader.n_search_axes
        self.best_pos = None
        self.best_cost = None

    def scale_value(self, val: float, min_val: float, max_val: float) -> float:
        return 2 * (val - min_val) / (max_val - min_val) - 1

    def unscale_value(self, val: float, min_val: float, max_val: float) -> float:
        return (1 + val) * (max_val - min_val) / 2 + min_val

    def unscale_design_point(self, position: np.ndarray) -> np.ndarray:
        if not isinstance(position, np.ndarray):
            raise ValueError("Query point to unscale_design_point must be a np array")
        unscaled_position = np.zeros_like(position)
        for i_axis in range(self.input_reader.n_search_axes):
            if self.axis_logscale[i_axis]:
                unscaled_position[i_axis] = 10 ** (
                    self.unscale_value(
                        position[i_axis],
                        self.min_search_axis[i_axis],
                        self.max_search_axis[i_axis],
                    )
                )
            else:
                unscaled_position[i_axis] = self.unscale_value(
                    position[i_axis],
                    self.min_search_axis[i_axis],
                    self.max_search_axis[i_axis],
                )
        return unscaled_position

    def run(self, file_obj: Path) -> tuple:
        """Run differential evolution and return (best_pos_scaled, best_cost).

        Uses vectorized=True so scipy passes the entire candidate population
        (shape: (popsize, n_params)) to the objective at once, letting
        _compute_all_losses dispatch via pmap just as PSO does.
        """
        bounds = [(-1.0, 1.0)] * self.n_search_axes
        n_pop = self.input_reader.population_size

        # generate initial population using the same LHS function as PSO
        lhs_bounds = np.array([[-1.0, 1.0]] * self.n_search_axes)
        initial_population = get_lhs_sampling(n_pop, lhs_bounds, seed=self.random_seed)

        print(f"DE population size: {n_pop}, max iterations: {self.input_reader.n_iters_pop}")
        with open(file_obj, 'a') as f:
            f.write(f"Population size: {n_pop}\n")
            f.write(f"Max iterations: {self.input_reader.n_iters_pop}\n\n")

        iteration_counter = [0]
        t_last = [time.time()]

        def callback(intermediate_result):
            iteration_counter[0] += 1
            it = iteration_counter[0]
            t_now = time.time()
            elapsed = t_now - t_last[0]
            t_last[0] = t_now
            cost = intermediate_result.fun
            print(f"Iteration: {it}")
            print(f"Best Cost: {cost:.4E}")
            print(f"Time for iteration: {elapsed:.4f}")
            print("-------")
            with open(file_obj, 'a') as f:
                f.write(f"{it}, {cost:.4E}, {elapsed:.4f}\n")
                f.flush()
            stop_flag = Path(self.input_reader.output_dir) / "stop_fitting.flag"
            if stop_flag.exists():
                return True  # signals DE to halt early

        def vectorized_objective(population: np.ndarray) -> np.ndarray:
            # scipy vectorized passes (n_params, m) after an internal .T; transpose to (m, n_params)
            return self.problem_obj._compute_all_losses(population.T)

        result = scipy.optimize.differential_evolution(
            vectorized_objective,
            bounds=bounds,
            maxiter=self.input_reader.n_iters_pop,
            init=initial_population,
            vectorized=True,
            seed=self.random_seed,
            callback=callback,
            mutation=(0.5, 1.0),
            recombination=0.7,
            tol=0,
            polish=False,  # gradient refinement is done separately by NODE
        )

        self.best_pos = np.array(result.x)
        self.best_cost = float(result.fun)
        print(f"Final best position (scaled): {self.best_pos}")
        print(f"Final best cost: {self.best_cost:.4E}")
        print("Done with DE search")
        return self.best_pos, self.best_cost
