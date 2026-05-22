import numpy as np

# Bonhoeffer et al. 1997, PNAS -- "Virus dynamics and drug therapy"
# Treatment phase model (protease inhibitor, beta=0): Eq. 1 with beta=0
#
# Ordering of parameters in trainable_parameters as provided by the user:
# ['a', 'u', 'k']
# Ordering of integrated variables as provided by the user:
# ['y', 'v']
#
# y: infected cells (cells/ml)
# v: free virus particles (copies/ml)  <-- this is the measured quantity
#
# Under protease inhibitor treatment (beta = 0):
#   dy/dt = -a * y
#   dv/dt =  k * y - u * v


def user_defined_system(t: float, y: np.ndarray, trainable_parameters: dict, fixed_parameters: dict, dataset: np.ndarray, t_eval: np.ndarray):

        a = trainable_parameters['a']
        u = trainable_parameters['u']
        k = trainable_parameters['k']

        y_cells = y[0]  # infected cells
        v_virus = y[1]  # free virus

        dy_dt = -a * y_cells
        dv_dt =  k * y_cells - u * v_virus

        return np.array([dy_dt, dv_dt])


def _compute_loss_problem(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):

        # dataset[:, 0] is viral load (copies/ml) -- the only measured quantity
        v_data = dataset[:, 0]
        v_sim  = solution[:, 1]

        # log-scale RMSE normalised by max log10 value
        eps = 1e-12
        log_data = np.log10(v_data + eps)
        log_sim  = np.log10(np.abs(v_sim) + eps)
        scale    = np.max(np.abs(log_data))

        loss = np.sqrt(np.mean(np.square((log_sim - log_data) / scale)))
        return loss


def writeout_description(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):

        # columns: time | v_data | v_sim | y_sim
        v_data = dataset[:, 0]
        v_sim  = solution[:, 1]
        y_sim  = solution[:, 0]

        Nts = solution_time.shape[0]
        writeout_array = np.zeros([Nts, 4])
        writeout_array[:, 0] = solution_time
        writeout_array[:, 1] = v_data
        writeout_array[:, 2] = v_sim
        writeout_array[:, 3] = y_sim

        return writeout_array
