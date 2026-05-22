import numpy as np

# Ordering of parameters in trainable_parameters as provided by the user:
# ['alpha', 'beta', 'gamma', 'cp', 'kp', 'de']
# Ordering of integrated variables as provided by the user:
# ['xp', 'vp', 'h']

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):

        alpha = trainable_parameters['alpha']
        beta  = trainable_parameters['beta']
        gamma = trainable_parameters['gamma']
        cp    = trainable_parameters['cp']
        kp    = trainable_parameters['kp']
        de    = trainable_parameters['de']
        mp    = fixed_parameters['mp']

        xp = y[0]
        vp = y[1]
        h  = y[2]

        def V_func(t):
                return 24 + 24 * np.sin(16 * np.pi * t)

        def V_dot_func(t):
                return 24 * 16 * np.pi * np.cos(16 * np.pi * t)

        V     = V_func(t)
        V_dot = V_dot_func(t)

        dxp_dt = vp
        dvp_dt = (kp * (de * V - h) - cp * vp - kp * xp) / mp
        dh_dt  = alpha * de * V_dot - beta * np.abs(V_dot) * np.abs(h) - gamma * V_dot * np.abs(h)

        return [dxp_dt, dvp_dt, dh_dt]


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        alpha = trainable_parameters['alpha']
        beta  = trainable_parameters['beta']
        gamma = trainable_parameters['gamma']
        cp    = trainable_parameters['cp']
        kp    = trainable_parameters['kp']
        de    = trainable_parameters['de']
        mp    = fixed_parameters['mp']

        exp_disp = dataset[:, 0]
        sim_disp = solution[:, 0]

        scale_factor = np.max(np.abs(exp_disp))
        loss = np.sqrt(np.mean(np.square((exp_disp - sim_disp) / scale_factor)))
        return loss


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        alpha = trainable_parameters['alpha']
        beta  = trainable_parameters['beta']
        gamma = trainable_parameters['gamma']
        cp    = trainable_parameters['cp']
        kp    = trainable_parameters['kp']
        de    = trainable_parameters['de']
        mp    = fixed_parameters['mp']

        def V_func(t):
                return 24 + 24 * np.sin(16 * np.pi * t)

        Nts = solution_time.shape[0]
        V_applied = np.zeros(Nts)
        for i in range(Nts):
                V_applied[i] = V_func(solution_time[i])

        writeout_array = np.zeros([Nts, 6])
        writeout_array[:, 0] = solution_time
        writeout_array[:, 1] = dataset[:, 0]   # experimental displacement
        writeout_array[:, 2] = solution[:, 0]  # simulated xp
        writeout_array[:, 3] = solution[:, 2]  # simulated h
        writeout_array[:, 4] = V_applied
        writeout_array[:, 5] = solution[:, 1]  # simulated vp
        return writeout_array
