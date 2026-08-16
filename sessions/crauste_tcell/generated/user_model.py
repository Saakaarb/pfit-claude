import numpy as np

# Ordering of parameters in trainable_parameters as provided by the user:
# ['delta_EL', 'delta_LM', 'delta_NE', 'mu_EE', 'mu_LE', 'mu_LL',
#  'mu_N', 'mu_P', 'mu_PE', 'mu_PL', 'rho_E', 'rho_P']
# Ordering of integrated variables as provided by the user:
# ['Naive', 'EarlyEffector', 'LateEffector', 'Memory', 'Pathogen']
#
# CD8 T-cell differentiation during an acute infection (Crauste et al. 2017).
# Naive cells are recruited by the pathogen into early effectors, which
# proliferate while pathogen is present and differentiate into late effectors
# and then into memory cells. Both effector compartments clear the pathogen.
# All death terms are density dependent (quadratic or bilinear), which is what
# makes the model's parameters only partially identifiable from the data.
#
# Data are staggered: each observable is measured at its own subset of the
# union time grid, so unmeasured cells arrive as NaN and are masked out of the
# loss (see lib/LLM/reference/staggered_data.md).


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):

        delta_EL = trainable_parameters['delta_EL']
        delta_LM = trainable_parameters['delta_LM']
        delta_NE = trainable_parameters['delta_NE']
        mu_EE    = trainable_parameters['mu_EE']
        mu_LE    = trainable_parameters['mu_LE']
        mu_LL    = trainable_parameters['mu_LL']
        mu_N     = trainable_parameters['mu_N']
        mu_P     = trainable_parameters['mu_P']
        mu_PE    = trainable_parameters['mu_PE']
        mu_PL    = trainable_parameters['mu_PL']
        rho_E    = trainable_parameters['rho_E']
        rho_P    = trainable_parameters['rho_P']

        N = y[0]
        E = y[1]
        L = y[2]
        M = y[3]
        P = y[4]

        # Naive: constant-rate loss, plus pathogen-driven recruitment into E
        dN_dt = -mu_N * N - delta_NE * N * P

        # Early effector: recruitment + pathogen-driven proliferation,
        # quadratic self-limitation, differentiation into L
        dE_dt = delta_NE * N * P + rho_E * E * P - mu_EE * E * E - delta_EL * E

        # Late effector: differentiation in, quadratic and E-mediated death,
        # differentiation into memory
        dL_dt = delta_EL * E - mu_LL * L * L - mu_LE * E * L - delta_LM * L

        # Memory: accumulates only, no loss on this timescale
        dM_dt = delta_LM * L

        # Pathogen: quadratic growth, linear clearance, killing by E and L
        dP_dt = rho_P * P * P - mu_P * P - mu_PE * E * P - mu_PL * L * P

        return np.array([dN_dt, dE_dt, dL_dt, dM_dt, dP_dt])


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        # dataset columns: 0-3 the four measured populations, 4-7 their
        # per-point standard deviations, in the same order. Time is not in
        # dataset. Cells not measured at a given time are NaN in both halves.
        measured = dataset[:, 0:4]
        sigma = dataset[:, 4:8]

        # The four observables are the first four states, in the same order.
        model_obs = solution[:, 0:4]

        # Sanitise BEFORE any arithmetic so NaN never enters the graph.
        mask = ~np.isnan(measured)
        data_safe = np.where(mask, measured, 0.0)
        sigma_safe = np.where(mask, sigma, 1.0)

        # Residuals in units of the measurement noise. The populations span
        # roughly four orders of magnitude across observables and time, so
        # noise weighting - not peak scaling - is what makes the four channels
        # comparable. A fit at the noise level scores ~1.
        resid = np.where(mask, (model_obs - data_safe) / sigma_safe, 0.0)
        loss = np.sqrt(np.sum(resid * resid) / np.sum(mask))

        return loss


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        Nts = solution_time.shape[0]

        # time | 4 measured | 4 simulated observables | simulated Pathogen
        writeout_array = np.zeros([Nts, 10])
        writeout_array[:, 0] = solution_time
        writeout_array[:, 1] = dataset[:, 0]   # measured Naive
        writeout_array[:, 2] = dataset[:, 1]   # measured EarlyEffector
        writeout_array[:, 3] = dataset[:, 2]   # measured LateEffector
        writeout_array[:, 4] = dataset[:, 3]   # measured Memory
        writeout_array[:, 5] = solution[:, 0]  # simulated Naive
        writeout_array[:, 6] = solution[:, 1]  # simulated EarlyEffector
        writeout_array[:, 7] = solution[:, 2]  # simulated LateEffector
        writeout_array[:, 8] = solution[:, 3]  # simulated Memory
        writeout_array[:, 9] = solution[:, 4]  # simulated Pathogen (unobserved)

        return writeout_array
