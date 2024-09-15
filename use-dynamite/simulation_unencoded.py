from dynamite.operators import sigmax, sigmay, sigmaz, index_sum
from dynamite.states import State
import numpy as np
import scipy as sp
import functools
import argparse
X, Y, Z = sigmax, sigmay, sigmaz


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, required=True,
                           help="random seed (used to sample the noise coefficients and the initial state)")
    argparser.add_argument("-n", type=int, required=True,
                           help="number of qubits")
    args = argparser.parse_args()

    if args.seed >= 0:
        seed_list = [args.seed]
    elif args.seed == -1:
        seed_list = list(range(50))
    else:
        print("invalid seed")
        exit(1)

    n = args.n

    t_list = 0.1 * np.arange(1,101)

    for t in t_list:
        for seed in seed_list:

            # build Hamiltonian
            H = index_sum(X(0), size=n) + index_sum(Z(0), size=n) + index_sum(Z(0) * Z(1), size=n)
            H.L = n

            # add coherent error
            np.random.seed(seed)
            epsx = np.random.uniform(-1,1,n)
            epsy = np.random.uniform(-1,1,n)
            epsz = np.random.uniform(-1,1,n)
            V = sum([epsx[i] * X(i) + epsy[i] * Y(i) + epsz[i] * Z(i) for i in range(n)])
            V.L = n

            # build noisy Hamiltonian
            H_noisy = H + V

            # randomly select an initial state
            psi0 = State(L=n, state='random', seed=seed)

            # evolve
            psi_true = H.evolve(psi0, t=t)
            psi_noisy = H_noisy.evolve(psi0, t=t)
            psi_true_np = psi_true.to_numpy()
            psi_noisy_np = psi_noisy.to_numpy()
            err = np.linalg.norm(psi_noisy_np - psi_true_np)

            # output
            f = open(f"sweep_time_unencoded_{n}qubits.txt", "a")
            f.write(f"#qubits = {n}, t = {t}, seed = {seed}, error = {err}\n")
            f.close()