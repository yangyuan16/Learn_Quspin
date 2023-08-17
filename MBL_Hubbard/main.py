from quspin.operators import hamiltonian, exp_op, quantum_operator # operators
from quspin.basis import spinful_fermion_basis_1d # Hilbert space basis
from quspin.tools.measurements import obs_vs_time # calculating dynamics
import numpy as np # general math function
from numpy.random import uniform, choice # tools for doing ramdoms sampling
from time import time 
import matplotlib.pyplot as plt # plotting library
#
# standard Fermi-Hubbard hamiltonian operator
def Hop_1d_OBC_standard(L, J, U, ):
    #
    # define site-coupling lists
    hop_right = [[-J, i, i+1] for i in range(L-1)] # hopping to the right OBC
    hop_left = [[J, i, i+1] for i in range(L-1)] # hopping to the left OBC
    int_list = [[U, i, i] for i in range(L)] # onsite interaction
    # create static lists
    operator_list_0 = [
        ["+-|", hop_left], # up hop left
        ["-+|", hop_right], # up hop right
        ["|+-", hop_left], # down hop left
        ["|-+", hop_right],  # down hop right
        ["n|n", int_list], # on site interaction
        ]
    return operator_list_0
#
# standard potential opertor term of Fermi-Hubbard 
def Hop_1d_potential_dict(L):
    # creating local potential operator dict
    op_pot_dict = {}
    for i in range(L):
        op_pot_dict["n"+str(i)] = [
            ["n|", [[1.0, i]]],
            ["|n", [[1.0, i]]]
        ]
    return op_pot_dict
# Hamiltonian operators combination
def Hop_combination(op_0,op_pot_dict):
    operator_h0_dict = {}
    operator_h0_dict["H0"] = op_0
    operator_dict = {**operator_h0_dict, **op_pot_dict} 
    return operator_dict
#
# define the  observable operators 定义观测量算符
def Obop_1d_imbalance_list(L, N): #
    # N: number of particles
    # L : lattice length
    # site-coupling lists to create the sublattice imbalance observable
    sublat_list = [[(-1.0)**i/N, i] for i in range(0, L)]
    imbalance_list = [
        ["n|", sublat_list],
        ["|n", sublat_list]
    ]
    return imbalance_list
#   
# define function to do dynamics for different disorder realizations.
# realize the time evolution
def real(H_dict,I,psi_0,w,t,i):
	# body of function goes below
	ti = time() # start timing function for duration of reach realisation
	# create a parameter list which specifies the onsite potential with disorder
	params_dict=dict(H0=1.0)
    # setting disorder
	for j in range(L):
		params_dict["n"+str(j)] = uniform(-w,w)
	# using the parameters dictionary construct a hamiltonian object with those
	# parameters defined in the list
	H = H_dict.tohamiltonian(params_dict)
	# use exp_op to get the evolution operator
	U = exp_op(H,a=-1j,start=t.min(),stop=t.max(),num=len(t),iterate=True)
	psi_t = U.dot(psi_0) # get generator psi_t for time evolved state
	# use obs_vs_time to evaluate the dynamics
	t = U.grid # extract time grid stored in U, and defined in exp_op
	obs_t = obs_vs_time(psi_t,t,dict(I=I))  # 计算观测量随着时间的变化
	# print reporting the computation time for realization
	print("realization {}/{} completed in {:.2f} s".format(i+1,n_real,time()-ti))
	# return observable values
	return obs_t["I"].real
#  
if __name__ == "__main__":
    #-----------------------------------------------------------#
    # physical parameter
    L = 8 # system size
    N = L // 2 # number of particles
    N_up = N // 2 + N % 2 # number of fermions with spin up
    N_down = N // 2 # number of fermions with spin down
    #
    J = 1.0 # hopping strength
    U = 1.0 # interaction strength
    #-----------------------------------------------------------#
    # create model
    # get standard 1d fermi-hubbard hamiltonian operators
    op_0_list = Hop_1d_OBC_standard(L=L, J=J, U=U)
    op_pot_dict = Hop_1d_potential_dict(L=8) # chemical potential operator
    # combine the standard fermi-hubbard hamiltonian operators with chemical potential operator
    op_dict = Hop_combination(op_0=op_0_list, op_pot_dict=op_pot_dict)
    # get hamiltonian dict
    basis = spinful_fermion_basis_1d(L=L, Nf=(N_up, N_down)) # build spinful fermions basis
    no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
    H_dict = quantum_operator(op_dict, basis=basis, **no_checks)
    #-------------------------------------------------------------#
    # set initial state
    # strings with represent the initial state 
    s_up = "".join("1000" for i in range(N_up))
    s_down = "".join("0010" for i in range(N_down))
    i_0 = basis.index(s_up, s_down) # find index of product state
    psi_0 = np.zeros(basis.Ns) # allocate space for state # len(list(basis))
    psi_0[i_0] = 1.0 # set MB state to be the given product state
    print("H-space size: {:d}, initial state: |{:s}>(x)|{:s}>".format(basis.Ns,s_up,s_down))
    #----------------------------------------------------------#
    # set observable operators imbalance
    # get imbalance list
    imbalance_list = Obop_1d_imbalance_list(L=L, N=N)
    I = hamiltonian(imbalance_list,[], basis=basis, **no_checks) # get imbalance operator
    #----------------------------------------------------------#
    # do the time evolution of hamiltonian under disorder
    # range in time to evolve system
    start, stop, num = 0.0, 35.0, 101
    t = np.linspace(start, stop, num=num, endpoint=True)
    w_list = [1.0, 4.0, 10.0] # disorder strength
    n_real = 100 # number of realizations
    n_boot = 100 # number of bootstrap samples to calculate error
    #  
    for w in w_list: #  loop for disorder strength
        I_data = []
        for it in range(n_real): # loop for disorder realization
            data = real(H_dict=H_dict, I=I, psi_0=psi_0,w=w,t=t,i=it)
            I_data.append(data)
        ### averaging and error estimation
        I_avg = np.mean(np.array(I_data), axis=0)
        # generate bootstrap samples
        bootstrap_gen = (np.array(I_data)[choice(n_real,size=n_real)].mean(axis=0) for i in range(n_boot))
        # generate the fluctuations about the mean of I
        sq_fluc_gen = ((bootstrap - I_avg)**2 for bootstrap in bootstrap_gen)
        I_error = np.sqrt(sum(sq_fluc_gen)/n_boot)
        ### plotting results
        plt.errorbar(t, I_avg, I_error, marker=".",label="w={:.2f}".format(w))

    # configuring plots
    plt.xlabel("$Jt$",fontsize=18)
    plt.ylabel("$\\mathcal{I}$",fontsize=18)
    plt.grid(True)
    plt.tick_params(labelsize=16)
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig('./figs/fermion_MBL.png', bbox_inches='tight')
    plt.show()
    plt.close()