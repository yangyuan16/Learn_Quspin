# MBL 项目，
from quspin.operators import quantum_operator # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import ent_entropy, diag_ensemble # entropies
from numpy.random import uniform,seed # pseudo random numbers
from joblib import delayed,Parallel # parallelisation
import numpy as np # generic math functions
from time import time # timing package
import matplotlib.pyplot as plt
#  define the standard XXZ chain
def HXXZ_1d_OBC_standard(L=10, Jxy=1.0, Jzz=1.0, hz=1.0):
    Jzz_chain = [[Jzz, i, i+1] for i in range(L-1)] # OBC
    Jxy_chain = [[Jxy/2.0, i, i+1] for i in range(L-1)] #OBC
    # define XXZ chain parameter
    op_dict = dict(Jxy=[["+-",Jxy_chain],["-+", Jxy_chain]], Jzz=[["zz", Jzz_chain]])
    # define operators for local disorderd field
    for i in range(L):
        op = [[hz,i]]
        op_dict["hz"+str(i)] = [["z", op]]
    return op_dict

# tune the hamiltonian parameters and diagonalize the hamiltonian
def eigen_HXXZ_by_tune_para(HXXZ,Jxy, Jzz, disorder_strength, seeds_num):
    N = H_XXZ.basis.N
    seed(seeds_num)
    hz_list = uniform(-1,1, size=N)
    paras = {"hz"+str(i): disorder_strength*hz_list[i] for i in range(N)}
    paras["Jxy"] = Jxy
    paras["Jzz"] = Jzz
    # diagonalize
    E, V = HXXZ.eigh(pars = paras)
    return E, V

# get the bandwith the eigen energy
def eigen_HXXZ_bandwith_by_tune_para(HXXZ,Jxy,Jzz,disorder_strength, seeds_num):
    N = H_XXZ.basis.N
    seed(seeds_num)
    hz_list = uniform(-1,1, size=N)
    paras = {"hz"+str(i): disorder_strength*hz_list[i] for i in range(N)}
    paras["Jxy"] = Jxy
    paras["Jzz"] = Jzz
    # diagonalize    
	# get many-body bandwidth at t=0
    eigsh_args=dict(k=2,which="BE",maxiter=1E4,return_eigenvectors=False,pars=paras)
    Emin,Emax=HXXZ.eigsh(**eigsh_args)
    return Emin, Emax
#
def eigen_HXXZ_findstate_by_tune_para(HXXZ, Jxy, Jzz, disorder_strength, sigma, seeds_num):
    N = H_XXZ.basis.N
    seed(seeds_num)
    hz_list = uniform(-1,1, size=N)
    paras = {"hz"+str(i): disorder_strength*hz_list[i] for i in range(N)}
    paras["Jxy"] = Jxy
    paras["Jzz"] = Jzz
    E, psi = HXXZ.eigsh(pars=paras, k=1, sigma=sigma, maxiter=1E4)
    return E, psi
#
# construct time-dependent equation
def ramp(t,v):
    return (0.5 + v*t)
# construct time-dependent hamiltonian   
def HXXZ_time_dependent_buildH(HXXZ, Jxy, v, disorder_strength, seeds_num):
    N = H_XXZ.basis.N
    seed(seeds_num)
    hz_list = uniform(-1,1,size=N)
    paras = {"hz"+str(i):disorder_strength * hz_list[i] for i in range(N)}
    paras["Jxy"] = Jxy
    paras["Jzz"] = (ramp, (v,))
    # construct time-dependent hamiltonian
    HXXZ_t = H_XXZ.tohamiltonian(pars=paras)
    return HXXZ_t
# time evolve
def HXXZ_time_dependent_evolve(HXXZ_t, psi_0, t0, tf):
    psi = HXXZ_t.evolve(psi_0, t0, tf)
    return psi
# 
def get_ent_entropy(basis, psi):
    subsys = range(basis.L//2) # define subsystem
    Sent = basis.ent_entropy(psi, sub_sys_A=subsys)["Sent_A"]
    return Sent
#
def get_diag_entropy(basis, psi, E, V, ):
    S_d = diag_ensemble(basis.L, psi, E, V,
                        Sd_Renyi=True)["Sd_pure"]
    return S_d
#
if __name__ == "__main__":
    ### define model parameters ###
    L = 10 # system size
    Jxy = 1.0 # xy interaction
    Jzz_0 = 1.0 # zz interaction 
    hz_0 = 1.0 #
    #------------------------------------------------------------#
    # define operators with OBC using site-coupling lists
    op_dict = HXXZ_1d_OBC_standard(L=L, Jxy=Jxy, Jzz=Jzz_0, hz=hz_0)
    #print("op_dict:\n",op_dict)
    # compute basis in the 0-total magnetisation sector (require L even)
    # compute hamiltonian
    basis = spin_basis_1d(L, m=0, pauli=False)
    H_XXZ = quantum_operator(op_dict, basis=basis, dtype=np.float64)
    #-------------------------------------------------------------#
    n_real = 100 # number of disorder relisations
    n_jobs = 2 # number of spawned processes used for parallelisation
    h_MBL = 3.9 # MBL disorder strength
    h_ETH = 0.1 # delocalised disorder strength
    #h_ds = h_MBL # disorder strength
    h_ds = h_ETH
    vs = np.logspace(-3.0,0.0,num=20,base=10) # log_2-spaced vector of ramp speeds
    #
    S_ent = []
    S_d = []
    #
    for it in range(n_real): # 对 disorder 进行循环, seeds_num 控制随机数以及无需的随机性
        ### define simulation parameters ###
        seeds_num = it
        ti = time()
        #---------------------------------------------------------------#
        Jzz_end = 1 # Jzz 随时间演化到最后为 1 
        E_end, V_end = eigen_HXXZ_by_tune_para(
            HXXZ=H_XXZ, Jxy=1.0, Jzz=Jzz_end, disorder_strength=h_ds, seeds_num=seeds_num)
        #E_end_ETH, V_end_ETH = eigen_HXXZ_by_tune_para(
        #    HXXZ=H_XXZ, Jxy=1.0, Jzz=Jzz_end, disorder_strength=h_ETH,seeds_num=seeds_num) 
        #---------------------------------------------------------------#
        Jzz_start = 0.5 # Jzz 的初始值为 0.5  
        Emin_start, Emax_start = eigen_HXXZ_bandwith_by_tune_para(
            HXXZ=H_XXZ,Jxy=1.0,Jzz=Jzz_start,disorder_strength=h_ds,seeds_num=seeds_num)
        # calculate middle of spectrum
        E_inf_temp = (Emax_start + Emin_start) / 2.0
        # get initial energy and initial state
        E_init, psi_init = eigen_HXXZ_findstate_by_tune_para(
            HXXZ=H_XXZ, Jxy=1.0, Jzz=Jzz_start, disorder_strength=h_ds,
            sigma=E_inf_temp,seeds_num=seeds_num)
        #----------------------------------------------------------------#
        psi_0 = psi_init.reshape((-1,))
        entropy_ent = []
        entropy_diag = []
        for v in vs:
            # 构造不同速度 v 的含时哈密顿量  
            H_t = HXXZ_time_dependent_buildH(HXXZ=H_XXZ, Jxy=1.0, v=v, 
                                             disorder_strength=h_ds, seeds_num=seeds_num)
            # 进行时间演化
            tf = 0.5 / v # (0.5 + v*t)Jzz_0,  
            psi = HXXZ_time_dependent_evolve(HXXZ_t=H_t,psi_0=psi_0,t0=0.0,tf=0.5/v)
            Ent_entropy = get_ent_entropy(basis=basis,psi=psi)
            Diag_entropy = get_diag_entropy(basis=basis, psi=psi, E=E_end, V=V_end)
            #
            entropy_ent.append(Ent_entropy)
            entropy_diag.append(Diag_entropy)
        S_ent.append(entropy_ent)
        S_d.append(entropy_diag)
        # show time taken
        print("disorder realization {}/{} cost time:".format(it+1,n_real), time() - ti)
        #print("entropy:\n", entropy)
    #----------------------------------------------------------------------#
    S_ent_mean = np.mean(np.array(S_ent),axis=0)
    S_d_mean = np.mean(np.array(S_d),axis=0)
    # using matplotlib plot S_ent and S_d
    fig, pltarr1 = plt.subplots(2, sharex=True) # define subplot panel
    # subplot1: diag entropy vs ramp speed
    pltarr1[0].plot(vs,S_d_mean,label=" ", marker=".",color="blue") # plot data
    pltarr1[0].set_ylabel("$s_d(t_f)$", fontsize=22) # label y-axis
    pltarr1[0].set_xlabel(r"$v/J_{zz}(0)$", fontsize=22) # label x-axis
    pltarr1[0].set_xscale("log") # set log scale on x-axis
    pltarr1[0].grid(True, which="both") # plot grid
    pltarr1[0].tick_params(labelsize=16)
    # subplot2: entanglement entropy vs ramp speed
    pltarr1[1].plot(vs,S_ent_mean, label=" ", marker=".",color="blue") # plot data
    pltarr1[1].set_ylabel(r"$s_{ent}(t_f)$", fontsize=22) # label y-axis
    pltarr1[1].set_xlabel(r"$v/J_{zz}(0)$", fontsize=22) # label x-axis
    pltarr1[1].set_xscale("log") # set log scale on x-axis
    pltarr1[1].grid(True, which="both") # plot grid
    pltarr1[1].tick_params(labelsize=16)
    # save figs
    fig.savefig("./figs/example1.png", bbox_inches="tight")
    #
    plt.show()


