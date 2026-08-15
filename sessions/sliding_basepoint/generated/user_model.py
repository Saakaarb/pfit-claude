import numpy as np

# Ordering of parameters in trainable_parameters as provided by the user:
# ['c2', 'Dk', 'Dc', 'm1', 'm2']
# Ordering of integrated variables as provided by the user:
# ['x1', 'x2', 'v1', 'v2', 'k', 'c1']


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):

        c2 = trainable_parameters['c2']
        Dk = trainable_parameters['Dk']
        Dc = trainable_parameters['Dc']
        m1 = trainable_parameters['m1']
        m2 = trainable_parameters['m2']
        vf = fixed_parameters['vf']

        x1 = y[0]
        x2 = y[1]
        v1 = y[2]
        v2 = y[3]
        k  = y[4]
        c1 = y[5]

        # coupling spring between the impacting mass and the sliding base
        Fs = k * (x2 - x1)

        # impacting mass: spring force less velocity-proportional damping
        dv1dt = (Fs - c1 * np.abs(v1) * np.sign(v1)) / m1

        # sliding base: stick-slip Coulomb friction. Stuck while the transmitted
        # force is below the threshold and the base is essentially at rest;
        # otherwise it slips against a constant opposing friction.
        F = -Fs
        stuck = np.logical_and(np.abs(F) < c2, np.abs(v2) < vf)
        F2 = np.where(stuck, 0.0, F - c2 * np.sign(v2))
        dv2dt = F2 / m2

        # stiffness and damping accumulate with dissipated mechanical power
        P = np.abs(m1 * v1 * dv1dt)
        dkdt  = Dk * P
        dc1dt = Dc * P

        return np.array([v1, v2, dv1dt, dv2dt, dkdt, dc1dt])


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        c2 = trainable_parameters['c2']
        m1 = trainable_parameters['m1']

        x1 = solution[:, 0]
        x2 = solution[:, 1]
        v1 = solution[:, 2]
        k  = solution[:, 4]
        c1 = solution[:, 5]

        # reconstruct the two measured channels from the trajectory
        Fs = k * (x2 - x1)
        F_sim = np.abs(Fs - c1 * np.abs(v1) * np.sign(v1))
        s_sim = x1

        # column 0 of the data is force in kN; the model works in newtons
        F_exp = 1000.0 * dataset[:, 0]
        s_exp = dataset[:, 1]

        # peak-normalised mean absolute error, equal weight on each channel, so
        # the O(1e5) N force residuals cannot swamp the O(1e-1) displacements
        loss_F = np.mean(np.abs(F_exp - F_sim)) / np.max(np.abs(F_exp))
        loss_s = np.mean(np.abs(s_exp - s_sim)) / np.max(np.abs(s_exp))

        return loss_F + loss_s


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        x1 = solution[:, 0]
        x2 = solution[:, 1]
        v1 = solution[:, 2]
        k  = solution[:, 4]
        c1 = solution[:, 5]

        Fs = k * (x2 - x1)
        F_sim = np.abs(Fs - c1 * np.abs(v1) * np.sign(v1))
        s_sim = x1

        F_exp = 1000.0 * dataset[:, 0]
        s_exp = dataset[:, 1]

        Nts = solution_time.shape[0]
        writeout_array = np.zeros([Nts, 5])
        writeout_array[:, 0] = solution_time
        writeout_array[:, 1] = F_exp     # measured force (N)
        writeout_array[:, 2] = s_exp     # measured displacement
        writeout_array[:, 3] = F_sim     # simulated force (N)
        writeout_array[:, 4] = s_sim     # simulated displacement
        return writeout_array
