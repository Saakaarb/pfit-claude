import numpy as np

# Key:
# Nts: number of time steps in dataset
# Ny: number of state variables defined in the user_input.xml file
# N_col: number of columns in dataset provided (including the first column as time)

# Rahman et al. (2016) HIV transmission model (PEtab benchmark).
# 7 compartments; time in years. Force of infection and stage-specific
# transmission rates are derived (SBML assignment rules) from the state and
# parameters, then used in the balances.
#
# Trainable parameter ordering:
#   0 infected_normal_transmission_rate_relative  (rel_n)
#   1 infected_moderate_transmission_rate         (beta_m)
#   2 infected_weak_transmission_rate_relative    (rel_w)
#   3 infected_weak_treatment_rate                (treat_w)
#   4 infected_normal_worsen_rate                 (worsen_n)
#   5 infected_moderate_worsen_rate               (worsen_m)
#   6 treated_moderate_improve_rate               (improve_m)
#   7 treated_weak_improve_rate                   (improve_w)
#   8 behavioural_change_rate                     (bcr)
#
# Integrated variable ordering (matches user_input.xml):
#   0 susceptible 1 infected_normal 2 infected_moderate 3 infected_weak
#   4 treated_normal 5 treated_moderate 6 treated_weak


def user_defined_system(t: float, y: np.ndarray, trainable_parameters: dict, fixed_parameters: dict, dataset: np.ndarray, t_eval: np.ndarray):

        # this part is to be populated by the model
        #--------------------------------
        rel_n     = trainable_parameters['infected_normal_transmission_rate_relative']
        beta_m    = trainable_parameters['infected_moderate_transmission_rate']
        rel_w     = trainable_parameters['infected_weak_transmission_rate_relative']
        treat_w   = trainable_parameters['infected_weak_treatment_rate']
        worsen_n  = trainable_parameters['infected_normal_worsen_rate']
        worsen_m  = trainable_parameters['infected_moderate_worsen_rate']
        improve_m = trainable_parameters['treated_moderate_improve_rate']
        improve_w = trainable_parameters['treated_weak_improve_rate']
        bcr       = trainable_parameters['behavioural_change_rate']

        recruitment = fixed_parameters['recruitment_rate']
        S_death     = fixed_parameters['susceptible_death_rate']
        In_death    = fixed_parameters['infected_normal_death_rate']
        Im_death    = fixed_parameters['infected_moderate_death_rate']
        Iw_death    = fixed_parameters['infected_weak_death_rate']
        Tn_death    = fixed_parameters['treated_normal_death_rate']
        Tm_death    = fixed_parameters['treated_moderate_death_rate']
        Tw_death    = fixed_parameters['treated_weak_death_rate']
        treat_n     = fixed_parameters['infected_normal_treatment_rate']
        treat_m     = fixed_parameters['infected_moderate_treatment_rate']
        t_factor    = fixed_parameters['treated_transmission_factor']

        susceptible      = y[0]
        infected_normal  = y[1]
        infected_moderate = y[2]
        infected_weak    = y[3]
        treated_normal   = y[4]
        treated_moderate = y[5]
        treated_weak     = y[6]
        #--------------------------------

        # this part is to be populated by the user
        #--------------------------------
        # Derived transmission rates (assignment rules)
        beta_n = rel_n * beta_m
        beta_w = rel_w * beta_m
        beta_t = t_factor * beta_m

        total_pop = (susceptible + infected_normal + infected_moderate + infected_weak
                     + treated_normal + treated_moderate + treated_weak)
        total_infected = (infected_normal + infected_moderate + infected_weak
                          + treated_normal + treated_moderate + treated_weak)

        # Force of infection with behavioural-change damping
        force_of_infection = ((beta_n * infected_normal + beta_m * infected_moderate
                               + beta_w * infected_weak
                               + beta_t * (treated_normal + treated_moderate + treated_weak))
                              / total_pop) * np.exp(-bcr * total_infected)

        dsusceptible_dt      = recruitment - force_of_infection * susceptible - S_death * susceptible
        dinfected_normal_dt  = (force_of_infection * susceptible
                                - worsen_n * infected_normal - treat_n * infected_normal
                                - In_death * infected_normal)
        dinfected_moderate_dt = (worsen_n * infected_normal
                                 - worsen_m * infected_moderate - treat_m * infected_moderate
                                 - Im_death * infected_moderate)
        dinfected_weak_dt    = (worsen_m * infected_moderate
                                - treat_w * infected_weak - Iw_death * infected_weak)
        dtreated_normal_dt   = (improve_m * treated_moderate + treat_n * infected_normal
                                - Tn_death * treated_normal)
        dtreated_moderate_dt = (improve_w * treated_weak - improve_m * treated_moderate
                                + treat_m * infected_moderate - Tm_death * treated_moderate)
        dtreated_weak_dt     = (treat_w * infected_weak - improve_w * treated_weak
                                - Tw_death * treated_weak)
        #--------------------------------

        derivatives = np.array([dsusceptible_dt, dinfected_normal_dt, dinfected_moderate_dt,
                                dinfected_weak_dt, dtreated_normal_dt, dtreated_moderate_dt,
                                dtreated_weak_dt])
        return derivatives  # of shape [Ny]. Each derivative term must be user defined


def _compute_loss_problem(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):

        loss = 0.0

        # this part is to be populated by the user
        #--------------------------------
        # Observable: prevalence (%) = (1 - susceptible / total_population) * 100
        total_pop = np.sum(solution, axis=1)
        prevalence = (1.0 - solution[:, 0] / total_pop) * 100.0

        prev_data = dataset[:, 0]
        scale = np.maximum(np.max(np.abs(prev_data)), 1e-12)
        # Noise is constant (sigma = 1) in the PEtab problem, so a peak-normalised
        # RMSE has the same minimiser as the Gaussian objective and stays ~[0,1].
        loss = np.sqrt(np.mean(np.square((prevalence - prev_data) / scale)))
        #--------------------------------

        return loss  # scalar


def writeout_description(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):

        writeout_array = np.zeros([solution_time.shape[0], 3])

        # this part is to be populated by the user
        #--------------------------------
        total_pop = np.sum(solution, axis=1)
        prevalence = (1.0 - solution[:, 0] / total_pop) * 100.0

        writeout_array[:, 0] = solution_time
        writeout_array[:, 1] = dataset[:, 0]    # data prevalence
        writeout_array[:, 2] = prevalence        # model prevalence
        #--------------------------------

        return writeout_array  # of custom shape
