import numpy as np

# Ordering of parameters in trainable_parameters as provided by the user:
# ['k1', 'k2', 'k3', 'k4']
# Ordering of integrated variables as provided by the user:
# ['x1', 'x2']
#
# Cascaded tanks with overflow (Schoukens et al., Nonlinear System
# Identification Benchmarks 2016). A pump drives water into an upper tank, which
# drains through a small opening into a lower tank, which drains back to the
# reservoir. Both outflows follow Bernoulli, hence the square roots.
#
#   dx1/dt = -k1*sqrt(x1) + k4*u(t)
#   dx2/dt =  k2*sqrt(x1) - k3*sqrt(x2)
#        y =  x2
#
# The pump voltage u(t) is an exogenous recorded signal, not an analytic
# function. It arrives as data column 2 and is interpolated onto the solver's own
# time points - the solver does not step on the 4 s data grid.
#
# The overflow is deliberately NOT modelled: this is the benchmark's own
# baseline. Expect systematic error where the record saturates at y = 10.


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):

        k1 = trainable_parameters['k1']
        k2 = trainable_parameters['k2']
        k3 = trainable_parameters['k3']
        k4 = trainable_parameters['k4']

        x1 = y[0]
        x2 = y[1]

        # dataset excludes the time column, so column 0 is the measured level
        # and column 1 is the pump voltage. Interpolate the input at t.
        u = np.interp(t, t_eval, dataset[:, 1])

        # Clamp before the square root. The measured levels stay well above zero,
        # but a candidate parameter set can drive a level negative, and
        # sqrt(negative) would put NaN into the loss for every later time point.
        sqrt_x1 = np.sqrt(np.maximum(x1, 0.0))
        sqrt_x2 = np.sqrt(np.maximum(x2, 0.0))

        dx1_dt = -k1 * sqrt_x1 + k4 * u
        dx2_dt = k2 * sqrt_x1 - k3 * sqrt_x2

        return np.array([dx1_dt, dx2_dt])


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        # One observable: the lower tank level, measured directly as x2.
        measured = dataset[:, 0]
        model_obs = solution[:, 1]

        # Peak-normalised RMSE. There is no uncertainty column in this record -
        # the benchmark quotes an output SNR of ~40 dB rather than per-point
        # sigmas - so the residual is scaled by the column's own peak (10.0).
        # Multiply the reported loss by that peak to recover the benchmark's
        # e_RMS in volts.
        scale = np.max(np.abs(measured))
        scale = np.where(scale == 0, 1.0, scale)

        loss = np.sqrt(np.mean(np.square((model_obs - measured) / scale)))

        return loss


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        Nts = solution_time.shape[0]

        # time | measured level | simulated level | simulated upper tank | input
        writeout_array = np.zeros([Nts, 5])
        writeout_array[:, 0] = solution_time
        writeout_array[:, 1] = dataset[:, 0]   # measured lower tank level
        writeout_array[:, 2] = solution[:, 1]  # simulated lower tank level
        writeout_array[:, 3] = solution[:, 0]  # simulated upper tank (unobserved)
        writeout_array[:, 4] = dataset[:, 1]   # pump voltage

        return writeout_array
