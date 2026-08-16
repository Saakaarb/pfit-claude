import numpy as np

# Ordering of parameters in trainable_parameters as provided by the user:
# ['eps1', 'eps2', 'q', 'f']
# Ordering of integrated variables as provided by the user:
# ['X', 'Y', 'Z']
#
# Oregonator (Field-Noyes) model of the Belousov-Zhabotinsky reaction in
# Tyson's dimensionless scaling. X is HBrO2, autocatalytic and fast; Y is
# bromide, faster still by the ratio eps1/eps2 and the variable that switches
# the autocatalysis off; Z is the oxidised catalyst, which relaxes on the slow
# O(1) timescale and feeds bromide back.
#
# The two small parameters set the stiffness directly: the X equation is scaled
# by 1/eps1 and the Y equation by 1/eps2, so the RHS spans 1 : 1/eps1 : 1/eps2,
# which over the search box reaches five orders of magnitude.
#
# Only X and Z are recorded. Y is unobserved.


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):

        eps1 = trainable_parameters['eps1']
        eps2 = trainable_parameters['eps2']
        q    = trainable_parameters['q']
        f    = trainable_parameters['f']

        X = y[0]
        Y = y[1]
        Z = y[2]

        # Autocatalytic production of X, quenched by Y and self-limited
        dX_dt = (q * Y - X * Y + X * (1.0 - X)) / eps1

        # Bromide: consumed by X, regenerated from the oxidised catalyst
        dY_dt = (-q * Y - X * Y + f * Z) / eps2

        # Catalyst relaxation on the slow timescale
        dZ_dt = X - Z

        return np.array([dX_dt, dY_dt, dZ_dt])


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        # dataset columns: 0-1 the measured X and Z, 2-3 their measurement
        # standard deviations. Time is not in dataset.
        measured = dataset[:, 0:2]
        sigma = dataset[:, 2:4]

        # X is state 0 and Z is state 2; Y (state 1) is not observed.
        model_obs = np.stack([solution[:, 0], solution[:, 2]], axis=1)

        # Residuals in units of the measurement noise. Both channels carry
        # constant noise, so this is the same as a peak-scaled RMSE up to the
        # factor 0.02, and a fit at the noise level scores ~1.
        resid = (model_obs - measured) / sigma
        loss = np.sqrt(np.mean(np.square(resid)))

        return loss


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        Nts = solution_time.shape[0]

        # time | 2 measured | 2 simulated observables | simulated Y
        writeout_array = np.zeros([Nts, 6])
        writeout_array[:, 0] = solution_time
        writeout_array[:, 1] = dataset[:, 0]   # measured X
        writeout_array[:, 2] = dataset[:, 1]   # measured Z
        writeout_array[:, 3] = solution[:, 0]  # simulated X
        writeout_array[:, 4] = solution[:, 2]  # simulated Z
        writeout_array[:, 5] = solution[:, 1]  # simulated Y (unobserved)

        return writeout_array
