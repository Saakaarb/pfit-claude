import numpy as np

# Key:
# Nts: number of time steps in dataset
# Ny: number of state variables defined in the user_input.xml file
# N_col: number of columns in dataset provided (including the first column as time)

# Sneyd & Dufour (2002) IP3-receptor gating (PEtab benchmark).
# 6-state Markov channel (O, R, I1, S, A, I2), conserved (sum = 1).
# Multi-experiment: 9 clamps differ only in the fixed IP3 and Ca concentrations,
# carried as two constant "input" states (ip3_in, ca_in) whose derivative is 0
# and whose initial values are set per experiment via INITIAL_CONDITIONS.
# dataset/t_eval represent a single experiment; the framework averages losses.
#
# Trainable parameter ordering:
#   0 k1  1 k2  2 k3  3 k4  4 k_1  5 k_2  6 k_3  7 k_4
#   8 l2  9 l4  10 l6  11 l_2  12 l_4  13 l_6
#
# Integrated variable ordering (matches user_input.xml):
#   0 IPR_O  1 IPR_R  2 IPR_I1  3 IPR_S  4 IPR_A  5 IPR_I2  6 ip3_in  7 ca_in


def user_defined_system(t: float, y: np.ndarray, trainable_parameters: dict, fixed_parameters: dict, dataset: np.ndarray, t_eval: np.ndarray):

        # this part is to be populated by the model
        #--------------------------------
        k1  = trainable_parameters['k1']
        k2  = trainable_parameters['k2']
        k3  = trainable_parameters['k3']
        k4  = trainable_parameters['k4']
        k_1 = trainable_parameters['k_1']
        k_2 = trainable_parameters['k_2']
        k_3 = trainable_parameters['k_3']
        k_4 = trainable_parameters['k_4']
        l2  = trainable_parameters['l2']
        l4  = trainable_parameters['l4']
        l6  = trainable_parameters['l6']
        l_2 = trainable_parameters['l_2']
        l_4 = trainable_parameters['l_4']
        l_6 = trainable_parameters['l_6']

        IPR_O  = y[0]
        IPR_R  = y[1]
        IPR_I1 = y[2]
        IPR_S  = y[3]
        IPR_A  = y[4]
        IPR_I2 = y[5]
        IP3    = y[6]   # constant per-experiment input
        Ca     = y[7]   # constant per-experiment input
        #--------------------------------

        # this part is to be populated by the user
        #--------------------------------
        # Derived equilibrium constants (SBML assignment rules)
        L1 = (k_1 * l2) / (k1 * l_2)
        L3 = (k_2 * l4) / (k2 * l_4)
        L5 = (k_4 * l6) / (k4 * l_6)

        # Transition fluxes (Sneyd Table 1 kinetic laws)
        v0 = ((k_2 + l_4 * Ca) / (1.0 + Ca / L5)) * IPR_O
        v1 = ((k2 * L3 + l4 * Ca) / (L3 + Ca * (1.0 + L3 / L1))) * IP3 * IPR_R
        v2 = (((k1 * L1 + l2) * Ca) / (L1 + Ca * (1.0 + L1 / L3))) * IPR_R
        v3 = (k_1 + l_2) * IPR_I1
        v4 = (((k4 * L5 + l6) * Ca) / (L5 + Ca)) * IPR_O
        v5 = ((L1 * (k_4 + l_6)) / (L1 + Ca)) * IPR_A
        v6 = (((k1 * L1 + l2) * Ca) / (L1 + Ca)) * IPR_A
        v7 = (k_1 + l_2) * IPR_I2
        v8 = ((k3 * L5) / (L5 + Ca)) * IPR_O
        v9 = k_3 * IPR_S

        dIPR_O_dt  = -v0 + v1 - v4 + v5 - v8 + v9
        dIPR_R_dt  =  v0 - v1 - v2 + v3
        dIPR_I1_dt =  v2 - v3
        dIPR_S_dt  =  v8 - v9
        dIPR_A_dt  =  v4 - v5 - v6 + v7
        dIPR_I2_dt =  v6 - v7
        dip3_in_dt = 0.0
        dca_in_dt  = 0.0
        #--------------------------------

        derivatives = np.array([dIPR_O_dt, dIPR_R_dt, dIPR_I1_dt, dIPR_S_dt, dIPR_A_dt,
                                dIPR_I2_dt, dip3_in_dt, dca_in_dt])
        return derivatives  # of shape [Ny]. Each derivative term must be user defined


def _compute_loss_problem(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):

        loss = 0.0

        # this part is to be populated by the user
        #--------------------------------
        # Observable: channel open probability Po = (0.9*A + 0.1*O)^4
        IPR_O = solution[:, 0]
        IPR_A = solution[:, 4]
        Po = np.power(0.9 * IPR_A + 0.1 * IPR_O, 4)

        Po_data = dataset[:, 0]
        # Po is a probability in [0,1]; a plain RMSE is already ~[0,1] and (with
        # the PEtab constant noise) has the same minimiser as the Gaussian loss.
        loss = np.sqrt(np.mean(np.square(Po - Po_data)))
        #--------------------------------

        return loss  # scalar


def writeout_description(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):

        writeout_array = np.zeros([solution_time.shape[0], 3])

        # this part is to be populated by the user
        #--------------------------------
        IPR_O = solution[:, 0]
        IPR_A = solution[:, 4]
        Po = np.power(0.9 * IPR_A + 0.1 * IPR_O, 4)

        writeout_array[:, 0] = solution_time
        writeout_array[:, 1] = dataset[:, 0]   # data open probability
        writeout_array[:, 2] = Po               # model open probability
        #--------------------------------

        return writeout_array  # of custom shape
