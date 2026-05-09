import numpy as np

# Key:
# Nts: number of time steps in dataset
# Ny: number of state variables defined in the user_input.xml file
# N_col: number of columns in dataset provided (including the first column as time)

# Ordering of parameters in trainable_parameters as provided by the user:
# ['Ea1', 'h1', 'A1', 'A2', 'Ea2', 'h2', 'm2', 'n2']
# Ordering of integrated variables as provided by the user:
# ['c1', 'c2', 'T']

def user_defined_system(t: float, y: np.ndarray, trainable_parameters: dict, fixed_parameters: dict, dataset: np.ndarray, t_eval: np.ndarray):

        # Arguments:
        # t: time (float)
        # y: state vector (np.ndarray of size [Ny])
        # trainable_parameters: dictionary of trainable parameters. keys identical to the names in the user_input.xml file
        # fixed_parameters: dictionary of fixed parameters. keys identical to the names in the user_input.xml file
        # dataset: dataset (np.ndarray of size [Nts,N_col-1]) (first dimension in dataset is time)
        # t_eval: evaluation times (np.ndarray of size [Nts])

        # this part is to be populated by the model
        #--------------------------------
        Ea1 = trainable_parameters['Ea1']
        h1  = trainable_parameters['h1']
        A1  = trainable_parameters['A1']
        A2  = trainable_parameters['A2']
        Ea2 = trainable_parameters['Ea2']
        h2  = trainable_parameters['h2']
        m2  = trainable_parameters['m2']
        n2  = trainable_parameters['n2']

        kb = fixed_parameters['kb']

        c1 = y[0]
        c2 = y[1]
        T  = y[2]

        #--------------------------------

        # this part is to be populated by the user
        #--------------------------------
        dc1_dt= - A1*np.exp(-Ea1/(kb*T))*c1
        dc2_dt= A2*np.exp(-Ea2/(kb*T))*c2**n2 * (1-c2)**m2
        dT_dt=np.abs(h1*dc1_dt)+np.abs(h2*dc2_dt)

        #if T > 500:

        #        dT_dt+=np.abs(h2*dc2_dt)
        
        #--------------------------------
        derivatives = np.array([dc1_dt, dc2_dt, dT_dt]) #
        return derivatives # of shape [Ny]. Each derivative term must be user defined

def _compute_loss_problem(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):

        # Arguments:
        # solution_time: time (np.ndarray of size [Nts])
        # solution: state vector (np.ndarray of size [Nts,Ny])
        # dataset: dataset (np.ndarray of size [Nts,N_col-1])
        # trainable_parameters: dictionary of trainable parameters. keys identical to the names in the user_input.xml file
        # fixed_parameters: dictionary of fixed parameters. keys identical to the names in the user_input.xml file

        loss = 0.0

        # this part is to be populated by the model
        #--------------------------------
        Ea1 = trainable_parameters['Ea1']
        h1  = trainable_parameters['h1']
        A1  = trainable_parameters['A1']
        A2  = trainable_parameters['A2']
        Ea2 = trainable_parameters['Ea2']
        h2  = trainable_parameters['h2']
        m2  = trainable_parameters['m2']
        n2  = trainable_parameters['n2']

        kb = fixed_parameters['kb']
        #--------------------------------

        # this part is to be populated by the user
        #--------------------------------
        Nts=solution_time.shape[0]
        heat_rate_pred=np.zeros(Nts)
        for i in range(Nts):
                derivs=user_defined_system(solution_time[i],solution[i,:],
                                                trainable_parameters,fixed_parameters)
                heat_rate_pred[i]=derivs[-1]
        # Add epsilon to avoid log10 of zero or negative
        eps = 1e-12
        loss1 = np.sqrt(np.mean(np.square((np.log10(heat_rate_pred+eps)-np.log10(dataset[:,-1]+eps))/np.log10(np.max(dataset[:,-1]+eps)))))
        loss2 = np.mean(np.abs((dataset[:,0]-solution[:,2])/np.max(np.abs(dataset[:,0]))))

        T_final_sim=solution[-1,-1]
        T_final_data=dataset[-1,0]

        if solution[-1,0] > 0.02 or solution[-1,1] < 0.98 or np.abs(T_final_sim-T_final_data)>50:

                loss3=500
        else:
                loss3=0
        #--------------------------------

        return loss1 + loss2 + loss3 # scalar


def writeout_description(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):

        # Arguments:
        # solution_time: time (np.ndarray of size [Nts])
        # solution: state vector (np.ndarray of size [Nts,Ny])
        # dataset: dataset (np.ndarray of size [Nts,N_col-1])
        # trainable_parameters: dictionary of trainable parameters. keys identical to the names in the user_input.xml file
        # fixed_parameters: dictionary of fixed parameters. keys identical to the names in the user_input.xml file
        # Change the size of writeout_array as per your requirement

        # this part is to be populated by the model
        #--------------------------------
        Ea1 = trainable_parameters['Ea1']
        h1  = trainable_parameters['h1']
        A1  = trainable_parameters['A1']
        A2  = trainable_parameters['A2']
        Ea2 = trainable_parameters['Ea2']
        h2  = trainable_parameters['h2']
        m2  = trainable_parameters['m2']
        n2  = trainable_parameters['n2']

        kb = fixed_parameters['kb']
        #--------------------------------

        writeout_array = np.zeros([solution_time.shape[0], 5]) # DEFINE THIS as required!

        # this part is to be populated by the user
        #--------------------------------
        writeout_array[:,0]=solution_time
        writeout_array[:,1]=dataset[:,0]
        writeout_array[:,2]=dataset[:,1]
        writeout_array[:,3]=solution[:,2]

        Nts = solution_time.shape[0]
        heat_rate_pred=np.zeros(Nts)
        for i in range(Nts):
                derivs=user_defined_system(solution_time[i],solution[i,:],
                                                trainable_parameters,fixed_parameters)
                heat_rate_pred[i]=derivs[-1]

        writeout_array[:,4]=heat_rate_pred
        #--------------------------------

        return writeout_array # of custom shape
