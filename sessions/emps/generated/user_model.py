import numpy as np

# Ordering of parameters in trainable_parameters as provided by the user:
# ['M', 'Fv', 'Fc', 'OF']
# Ordering of integrated variables as provided by the user:
# ['q', 'v']
#
# Electro-Mechanical Positioning System (Janot, Gautier & Brunot 2019): a mass
# driven along a linear axis by a motor, against viscous and dry friction.
#
#   dq/dt   = v
#   M*dv/dt = gtau*vir(t) - Fv*v - Fc*sign(v) - OF
#
# M is the total inertia seen at the load side, Fv the viscous coefficient, Fc
# the Coulomb (dry) friction force and OF a constant force offset. gtau converts
# the recorded motor voltage into a force at the load side.
#
# The motor voltage is an exogenous recorded signal (data column 2), interpolated
# onto the solver's own time points.
#
# The dry-friction switch is REGULARISED: tanh(v / v_eps) in place of sign(v).
#
# Why, per R1 in generated/user_input_check.txt: sign(v) is a chattering
# discontinuity (the velocity reverses 7 times here, re-crossing it each time),
# and the box also admits a slaved fast mode, since only the ratio Fv/M enters
# the Jacobian and that ratio spans eight decades. Stiff AND chattering is the
# one case where neither solver family works, and R1's instruction there is to
# fix the model rather than pick a lesser evil.
#
# The regularisation converts non-smoothness into stiffness, which an implicit
# solver can absorb. v_eps is the measurement's own velocity resolution
# (position quantum ~1e-07 m over a 1 ms sample), so the smoothing width is
# physically meaningful rather than a numerical fudge: below it, the encoder
# cannot distinguish motion from rest anyway. The cost is that the friction
# term now contributes its own timescale, M*v_eps/Fc ~ 5e-04 s at the reference
# values - which is why the integrator is implicit.
#
# This departs from the published reference model, so parameters recovered here
# are comparable to Janot et al. only approximately.


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):

        M  = trainable_parameters['M']
        Fv = trainable_parameters['Fv']
        Fc = trainable_parameters['Fc']
        OF = trainable_parameters['OF']

        gtau = fixed_parameters['gtau']
        v_eps = fixed_parameters['v_eps']

        q = y[0]
        v = y[1]

        # dataset excludes the time column, so column 0 is the measured position
        # and column 1 is the motor voltage. Interpolate the input at t.
        vir = np.interp(t, t_eval, dataset[:, 1])

        # Applied force at the load side, less viscous drag, dry friction and a
        # constant offset. tanh(v/v_eps) is the regularised sign: it saturates
        # to +/-1 within a few v_eps of zero and is smooth through it.
        force = gtau * vir - Fv * v - Fc * np.tanh(v / v_eps) - OF

        dq_dt = v
        dv_dt = force / M

        return np.array([dq_dt, dv_dt])


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        # One observable: the axis position, measured directly as q.
        measured = dataset[:, 0]
        model_obs = solution[:, 0]

        # Peak-normalised RMSE. The record carries no uncertainty column, so the
        # residual is scaled by the position column's own peak (0.2464 m).
        # Multiply the reported loss by that peak to recover an RMSE in metres.
        scale = np.max(np.abs(measured))
        scale = np.where(scale == 0, 1.0, scale)

        loss = np.sqrt(np.mean(np.square((model_obs - measured) / scale)))

        return loss


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        Nts = solution_time.shape[0]

        # time | measured position | simulated position | simulated velocity | input
        writeout_array = np.zeros([Nts, 5])
        writeout_array[:, 0] = solution_time
        writeout_array[:, 1] = dataset[:, 0]   # measured position
        writeout_array[:, 2] = solution[:, 0]  # simulated position
        writeout_array[:, 3] = solution[:, 1]  # simulated velocity (unobserved)
        writeout_array[:, 4] = dataset[:, 1]   # motor voltage

        return writeout_array
