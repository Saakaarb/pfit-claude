import numpy as np

# Ordering of parameters in trainable_parameters as provided by the user:
# ['k1', 'k2', 'k3', 'k4', 'k_1', 'k_2', 'k_3', 'k_4',
#  'l2', 'l4', 'l6', 'l_2', 'l_4', 'l_6']
# Ordering of integrated variables as provided by the user:
# ['O', 'R', 'I1', 'S', 'A', 'I2', 'IP3', 'Ca']
#
# Six-state IP3-receptor gating model of Sneyd & Dufour (2002). IP3 and Ca are
# clamped per experiment, so they are carried as extra states with zero
# derivative; their per-experiment values come from initial_conditions in the config.


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
        # dataset and t_eval represent ONE experiment's data; the framework
        # calls this function once per experiment

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

        O   = y[0]
        R   = y[1]
        I1  = y[2]
        S   = y[3]
        A   = y[4]
        I2  = y[5]
        IP3 = y[6]
        Ca  = y[7]

        # equilibrium constants derived from the fitted rates
        L1 = k_1 * l2 / (k1 * l_2)
        L3 = k_2 * l4 / (k2 * l_4)
        L5 = k_4 * l6 / (k4 * l_6)

        # the ten transition fluxes
        v0 = (k_2 + l_4 * Ca) / (1.0 + Ca / L5) * O                              # O  -> R
        v1 = (k2 * L3 + l4 * Ca) / (L3 + Ca * (1.0 + L3 / L1)) * IP3 * R         # R  -> O
        v2 = (k1 * L1 + l2) * Ca / (L1 + Ca * (1.0 + L1 / L3)) * R               # R  -> I1
        v3 = (k_1 + l_2) * I1                                                    # I1 -> R
        v4 = (k4 * L5 + l6) * Ca / (L5 + Ca) * O                                 # O  -> A
        v5 = L1 * (k_4 + l_6) / (L1 + Ca) * A                                    # A  -> O
        v6 = (k1 * L1 + l2) * Ca / (L1 + Ca) * A                                 # A  -> I2
        v7 = (k_1 + l_2) * I2                                                    # I2 -> A
        v8 = k3 * L5 / (L5 + Ca) * O                                             # O  -> S
        v9 = k_3 * S                                                             # S  -> O

        dOdt  = -v0 + v1 - v4 + v5 - v8 + v9
        dRdt  = v0 - v1 - v2 + v3
        dI1dt = v2 - v3
        dSdt  = v8 - v9
        dAdt  = v4 - v5 - v6 + v7
        dI2dt = v6 - v7

        # clamped inputs: held constant over the solve
        dIP3dt = 0.0
        dCadt = 0.0

        return np.array([dOdt, dRdt, dI1dt, dSdt, dAdt, dI2dt, dIP3dt, dCadt])


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
        # dataset and t_eval represent ONE experiment's data; the framework
        # calls this function once per experiment and averages the results

        O = solution[:, 0]
        A = solution[:, 4]

        # the measured observable is the channel open probability
        Po_sim = (0.9 * A + 0.1 * O) ** 4
        Po_exp = dataset[:, 0]

        # Po is already a probability in [0, 1], so a plain RMSE is normalized
        return np.sqrt(np.mean(np.square(Po_sim - Po_exp)))


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        O = solution[:, 0]
        A = solution[:, 4]

        Po_sim = (0.9 * A + 0.1 * O) ** 4
        Po_exp = dataset[:, 0]

        Nts = solution_time.shape[0]
        writeout_array = np.zeros([Nts, 3])
        writeout_array[:, 0] = solution_time
        writeout_array[:, 1] = Po_exp     # measured open probability
        writeout_array[:, 2] = Po_sim     # simulated open probability
        return writeout_array
