# Theophylline 1-compartment oral pharmacokinetic model
# Data: R Theoph dataset, Subject 6 (Wt=80 kg, Dose=4 mg/kg)
# Source: Boeckmann, Sheiner & Beal (1994); Upton study
#
# ODE system:
#   dA_gut/dt    = -ka * A_gut
#   dA_plasma/dt =  ka * A_gut - ke * A_plasma
#
# Observed: C_plasma = A_plasma / Vd  [mg/L]
# Units: amounts in mg, time in hours, Vd in L
#
# Published population PK values:
#   ka  ~ 1.5   h^-1
#   ke  ~ 0.086 h^-1
#   Vd  ~ 37.6  L  (= 0.47 L/kg * 80 kg for Subject 6)
#
# NOTE: dataset has 1 column (C_plasma_obs); solution has 2 states (A_gut, A_plasma).
# The loss function maps A_plasma / Vd → C_pred for comparison.

# Ordering of parameters in trainable_parameters as provided by the user:
# ['ka', 'ke', 'Vd']
# Ordering of integrated variables as provided by the user:
# ['A_gut', 'A_plasma']

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    ka = trainable_parameters['ka']
    ke = trainable_parameters['ke']

    A_gut    = y[0]
    A_plasma = y[1]

    dA_gut_dt    = -ka * A_gut
    dA_plasma_dt =  ka * A_gut - ke * A_plasma

    return np.array([dA_gut_dt, dA_plasma_dt])


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    Vd = trainable_parameters['Vd']

    C_pred = solution[:, 1] / Vd     # predicted plasma concentration [mg/L]
    C_obs  = dataset[:, 0]            # observed plasma concentration [mg/L]

    # normalize by max observed concentration to keep loss O(1)
    scale = np.max(np.abs(C_obs))
    scale = np.where(scale == 0, 1.0, scale)
    return np.sqrt(np.mean(np.square((C_pred - C_obs) / scale)))


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    Vd = trainable_parameters['Vd']
    Nts = solution_time.shape[0]
    writeout_array = np.zeros([Nts, 5])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]       # C_obs  [mg/L]
    writeout_array[:, 2] = solution[:, 1] / Vd # C_pred [mg/L]
    writeout_array[:, 3] = solution[:, 0]       # A_gut  [mg]
    writeout_array[:, 4] = solution[:, 1]       # A_plasma [mg]
    return writeout_array
