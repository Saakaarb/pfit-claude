# Lotka-Volterra predator-prey model
# Data: Hudson's Bay Company lynx-hare pelt counts, 1900-1920
# Source: http://www.math.tamu.edu/~phoward/m442/modbasics.pdf
#
# ODE system:
#   dH/dt = alpha * H - beta * H * L       (hare)
#   dL/dt = delta * H * L - gamma * L      (lynx)
#
# Units: populations in thousands, time in years
# Both states are directly observed in the CSV.
#
# Published fit (Stan, Carpenter 2018):
#   alpha = 0.55 yr^-1   (hare birth rate)
#   beta  = 0.028        (predation rate)
#   delta = 0.024        (prey-to-predator conversion)
#   gamma = 0.80 yr^-1   (lynx death rate)

# Ordering of parameters in trainable_parameters as provided by the user:
# ['alpha', 'beta', 'delta', 'gamma']
# Ordering of integrated variables as provided by the user:
# ['H', 'L']

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    alpha = trainable_parameters['alpha']
    beta  = trainable_parameters['beta']
    delta = trainable_parameters['delta']
    gamma = trainable_parameters['gamma']

    H = y[0]
    L = y[1]

    dHdt = alpha * H - beta * H * L
    dLdt = delta * H * L - gamma * L

    return np.array([dHdt, dLdt])


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    # Both H and L are observed; normalize by their typical scale (~50 thousand)
    scale = np.max(dataset, axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return np.sqrt(np.mean(np.square((solution - dataset) / scale)))


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    Nts = solution_time.shape[0]
    writeout_array = np.zeros([Nts, 5])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1:3] = dataset      # observed H, L
    writeout_array[:, 3:5] = solution     # predicted H, L
    return writeout_array
