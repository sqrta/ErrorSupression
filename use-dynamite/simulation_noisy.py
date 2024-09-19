from dynamite.operators import sigmax, sigmay, sigmaz, index_sum
from dynamite.states import State
import numpy as np
import scipy as sp
import functools
import argparse
X, Y, Z = sigmax, sigmay, sigmaz
MAX_ITS = 100000

def array2state(L, arr):
    """
    Convert a numpy array (in little-endian) to an L-qubit state.
    """
    assert 2**L == len(arr)
    s = State(L=L)
    s.set_all_by_function(lambda i: arr[i])
    return s

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, required=True,
                           help="random seed (used to sample the noise coefficients and the initial state)")
    argparser.add_argument("-nb", type=int, required=True,
                           help="number of blocks")
    argparser.add_argument("--lamb", type=int, required=True,
                           help="penalty coefficient")
    argparser.add_argument("-t", type=float, required=True,
                           help="evolving time")
    args = argparser.parse_args()

    if args.seed >= 0:
        seed_list = [args.seed]
    elif args.seed == -1:
        seed_list = list(range(50))
    else:
        print("invalid seed")
        exit(1)

    if args.lamb >= 0:
        lamb_list = [args.lamb]
    elif args.lamb == -1:
        lamb_list = 2 ** np.arange(5,15)
    else:
        print("invalid lamb")
        exit(1)

    if args.t > 0:
        t_list = [args.t]
    else:
        t_list = 0.1 * np.arange(1,101)

    nb = args.nb

    for lamb in lamb_list:
        for seed in seed_list:
            for t in t_list:
                nl = 2 * nb # number of logical qubits
                n = 4 * nb # number of physical qubits
                
                # build target Hamiltonian
                Htar = index_sum(X(0), size=nl) + index_sum(Z(0), size=nl) + index_sum(Z(0) * Z(1), size=nl)
                Htar.L = nl
            
                # build penalty Hamiltonian
                Hpen = sum([X(2*i) * X(2*i+1) + 3 * Z(2*i) * Z(2*i+1) for i in range(2 * nb)])
                Hpen.L = n
                
                # build encoding Hamiltonian
                Henc = sum([X(4*i)*X(4*i+1) - X(4*i)*X(4*i+2) for i in range(nb)]) # single logical X
                Henc += sum([Z(4*i)*Z(4*i+1) + Z(4*i)*Z(4*i+2) for i in range(nb)]) # single logical Z
                Henc += sum([Z(4*i+1)*Z(4*i+2) for i in range(nb)]) # in-block logical ZZ
                Henc += sum([Z(4*i+4)*Z(4*i+5) + np.sqrt(8*lamb/3) * (Z(4*i+1)*X(4*i+6) + Z(4*i+3)*X(4*i+6)) for i in range(nb-1)]) # cross-block logical ZZ
                Henc.L = n
            
                # build simulator Hamiltonian
                Hsim = lamb * Hpen + Henc
            
                # add coherent error
                np.random.seed(seed)
                epsx = np.random.uniform(-1,1,n)
                epsy = np.random.uniform(-1,1,n)
                epsz = np.random.uniform(-1,1,n)
                V = sum([epsx[i] * X(i) + epsy[i] * Y(i) + epsz[i] * Z(i) for i in range(n)])
                V.L = n
            
                # build noisy total Hamiltonian
                Htot = Hsim + V
                
                # logical states per block
                v00 = np.sqrt(0.5) * (State(state='0001') - State(state='1110'))
                v10 = np.sqrt(0.5) * (State(state='0100') - State(state='1011'))
                v01 = np.sqrt(0.5) * (State(state='1101') - State(state='0010'))
                v11 = np.sqrt(0.5) * (State(state='1000') - State(state='0111'))
                v00_np = v00.to_numpy()
                v10_np = v10.to_numpy()
                v01_np = v01.to_numpy()
                v11_np = v11.to_numpy()
                # encoder isometry per block
                U0 = sp.sparse.csc_matrix(np.column_stack((v00_np, v10_np, v01_np, v11_np)))
                # overall encoder isometry
                U = functools.reduce(sp.sparse.kron, [U0] * nb)
            
                # randomly select an initial state
                psi0 = State(L=nl, state='random', seed=seed) # lower case: unencoded states
                psi0_np = psi0.to_numpy()
                PSI0_np = U @ psi0_np
                PSI0 = array2state(L=n, arr=PSI0_np) # UPPER case: encoded states
        
                # evolve
                psi = Htar.evolve(psi0, t=t)
                PSI = Htot.evolve(PSI0, t=t, max_its=MAX_ITS)
                psi_np = psi.to_numpy()
                PSI_np = PSI.to_numpy()
                err = np.linalg.norm(PSI_np - U @ psi_np)

                # output
                f = open(f"output_noisy_{nb}blocks_lamb{lamb}_seed{seed}_sweep_time.txt", "a")
                # f = open(f"output_noisy_{nb}blocks_sweep_lamb.txt", "a")
                f.write(f"#blocks = {nb}, lamb = {lamb}, t = {t}, seed = {seed}, error = {err}\n")
                f.close()