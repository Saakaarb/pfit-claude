import numpy as np


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    # Trainable (order matches XML): beta1, beta2, k1, k2, a
    beta1 = trainable_parameters["beta1"]
    beta2 = trainable_parameters["beta2"]
    k1    = trainable_parameters["k1"]
    k2    = trainable_parameters["k2"]
    a     = trainable_parameters["a"]

    lam = fixed_parameters["lam"]
    d   = fixed_parameters["d"]
    u   = fixed_parameters["u"]
    mu  = fixed_parameters["mu"]

    x_cells = y[0]   # uninfected CD4+ cells
    y1_inf  = y[1]   # cells infected by drug-sensitive virus
    y2_inf  = y[2]   # cells infected by drug-resistant virus
    v1_free = y[3]   # drug-sensitive free virus
    v2_free = y[4]   # drug-resistant free virus

    dx  = lam - d*x_cells - beta1*x_cells*v1_free - beta2*x_cells*v2_free
    dy1 = beta1*(1 - mu)*x_cells*v1_free + beta2*mu*x_cells*v2_free - a*y1_inf
    dy2 = beta1*mu*x_cells*v1_free + beta2*(1 - mu)*x_cells*v2_free - a*y2_inf
    dv1 = k1*y1_inf - u*v1_free
    dv2 = k2*y2_inf - u*v2_free

    return np.array([dx, dy1, dy2, dv1, dv2])


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    # dataset col 0: CD4 count x; col 1: total viral load V = v1+v2
    x_data = dataset[:, 0]
    V_data = dataset[:, 1]

    x_sim  = solution[:, 0]
    V_sim  = solution[:, 3] + solution[:, 4]  # v1 + v2

    eps = 1e-12

    # Log-scale RMSE for viral load (spans many orders of magnitude)
    log_V_data = np.log10(V_data + eps)
    log_V_sim  = np.log10(np.abs(V_sim) + eps)
    scale_V    = np.max(np.abs(log_V_data))
    loss_V     = np.sqrt(np.mean(np.square((log_V_sim - log_V_data) / scale_V)))

    # Log-scale RMSE for CD4 count
    log_x_data = np.log10(x_data + eps)
    log_x_sim  = np.log10(np.abs(x_sim) + eps)
    scale_x    = np.max(np.abs(log_x_data))
    loss_x     = np.sqrt(np.mean(np.square((log_x_sim - log_x_data) / scale_x)))

    return loss_V + loss_x


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    x_data = dataset[:, 0]
    V_data = dataset[:, 1]

    x_sim  = solution[:, 0]
    V_sim  = solution[:, 3] + solution[:, 4]

    Nts = solution_time.shape[0]
    out = np.zeros((Nts, 5))
    out[:, 0] = solution_time
    out[:, 1] = x_data
    out[:, 2] = x_sim
    out[:, 3] = V_data
    out[:, 4] = V_sim

    return out
