# Ordering of parameters in trainable_parameters as provided by the user:
# ['alpha', 'beta', 'gamma', 'mp', 'cp', 'kp', 'de']
# Ordering of integrated variables as provided by the user:
# ['xp', 'vp', 'h']

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset):
        # Trainable parameters
        alpha = trainable_parameters['alpha']
        beta = trainable_parameters['beta']
        gamma = trainable_parameters['gamma']
        mp = fixed_parameters['mp']
        cp = trainable_parameters['cp']
        kp = trainable_parameters['kp']
        de = trainable_parameters['de']
        # Fixed parameters
        # Integrable variables (in order as provided by user)
        xp = y[0]  # xp is at index 0
        vp = y[1]  # vp is at index 1
        h = y[2]  # h is at index 2
    
        def V_func(t):
            return 24+24*np.sin(16*np.pi*t)
    
        def V_dot_func(t):
            return 24*16*np.pi* np.cos(16*np.pi * t)
    
        V = V_func(t)
    	V_dot = V_dot_func(t)
        
        dxp_dt = vp
    	dvp_dt = (kp * (de * V - h) - cp * vp - kp * xp) / mp
    	dh_dt = alpha * de * V_dot - beta * np.abs(V_dot) * np.abs(h) - gamma * V_dot * np.abs(h)
    
        # ... your code here ...
    
        return [dxp_dt, dvp_dt, dh_dt]

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
        # Trainable parameters
        alpha = trainable_parameters['alpha']
        beta = trainable_parameters['beta']
        gamma = trainable_parameters['gamma']
        mp = fixed_parameters['mp']
        cp = trainable_parameters['cp']
        kp = trainable_parameters['kp']
        de = trainable_parameters['de']
        # Fixed parameters
    
        exp_disp=dataset[:,0]
        sim_disp=solution[:,0]
    
        scale_factor=np.max(np.abs(exp_disp))
        loss=np.sqrt(np.mean(np.square((exp_disp-sim_disp)/scale_factor)))
        return loss

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

    alpha = trainable_parameters[&amp;amp;amp;#39;alpha&amp;amp;amp;#39;]
    beta = trainable_parameters[&amp;amp;amp;#39;beta&amp;amp;amp;#39;]
    gamma = trainable_parameters[&amp;amp;amp;#39;gamma&amp;amp;amp;#39;]
    mp = fixed_parameters[&amp;amp;amp;#39;mp&amp;amp;amp;#39;]
    cp = trainable_parameters[&amp;amp;amp;#39;cp&amp;amp;amp;#39;]
    kp = trainable_parameters[&amp;amp;amp;#39;kp&amp;amp;amp;#39;]
    de = trainable_parameters[&amp;amp;amp;#39;de&amp;amp;amp;#39;]
    xp = solution[:, 0]  # xp is at column 0
    vp = solution[:, 1]  # vp is at column 1
    h = solution[:, 2]  # h is at column 2

	def V_func(t):
        return 24+24*np.sin(16*np.pi*t)
                              
    # Change the size of writeout_array as per your requirement
    Nts = solution_time.shape[0]
    writeout_array = np.zeros([Nts, 6])
	V_applied=np.zeros(Nts)
    for i in range(Nts):
    	V_applied[i]=V_func(solution_time[i])
                              
    writeout_array[:,0]=solution_time
    writeout_array[:,1]=dataset[:,0]
    writeout_array[:,2]=solution[:,0]
    writeout_array[:,3]=solution[:,2]
    writeout_array[:,4]=V_applied
    writeout_array[:,5]=solution[:,1]
    return writeout_array