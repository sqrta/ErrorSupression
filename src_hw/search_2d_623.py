# %%
from Hamil_search1 import *
from simulation import blocks2Mat
import numpy as np
import scipy.sparse as sp

# %%
def zero_eigen_projector(H, tol=1e-10):
    """
    Compute the projector onto the zero-eigenvalue subspace of sparse matrix H.

    Parameters
    ----------
    H : scipy.sparse matrix or ndarray
        The (square) matrix whose zero-eigenvalue projector to compute.
    tol : float, optional
        Numerical tolerance to consider eigenvalues as zero.

    Returns
    -------
    P : np.ndarray
        The dense projector matrix onto the null space of H.
    """
    # Convert to dense if small enough; otherwise use sparse solver
    n = H.shape[0]
    if sp.issparse(H):
        H = H.tocsc()

    # Try to find all eigenvalues (dense path if small)

    H_dense = H.toarray() if sp.issparse(H) else H
    eigvals, eigvecs = np.linalg.eigh(H_dense)
    # a = np.array([1 if abs(x) < 0.5 else 0 for x in eigvals])
    # P0 = eigvecs @ (np.expand_dims(a, axis=1) * eigvecs.conj().T)
    b = np.array([1/x if abs(x) > 0.5 else 0 for x in eigvals])
    HpenInverse = eigvecs @ (np.expand_dims(b, axis=1) * eigvals.conj().T)

    # Find indices of (numerically) zero eigenvalues
    zero_idx = np.where(np.abs(eigvals) < tol)[0]

    if len(zero_idx) == 0:
        # No zero eigenvalues → zero projector
        print("No zero eigenvalues")
        return np.zeros((n, n), dtype=complex), None

    # Form projector P = sum_i v_i v_i†
    eigvecs_zero = eigvecs[:, zero_idx]
    print(f"eigvecs_zero shape {eigvecs_zero.shape}")
    P = eigvecs_zero @ eigvecs_zero.conj().T

    return P, HpenInverse

# %%
# blockSize = 6
# m = n // 2
# Xlist = ["X0*X1", "-1*X2*X3"]
# Zlist = ["2*Z2*Z3", "-2*Z4*Z5"]
# Ylist = [f"-{int(2**(m-1))}*Y0*Y1", f"{int(2**(m-1))}*Y{n-2}*Y{n-1}"]
# Hlist = Xlist + Ylist + Zlist
# print(Hlist)

def getHpen(blockNum):
    """
    `blockSize` means number of physical qubits in each block.

    e.g. `getHpen(8,2)` returns X0X1+3Z0Z1+X2X3+3Z2Z3+X4X5+3Z4Z5+X6X7+3Z6Z7
    """
    blockSize = 6
    Hpen_block = []
    m = blockSize // 2
    for i in range(blockNum):
        Hpen_block += [(1, f'X{6*i}*X{6*i+1}'), (2, f'Z{6*i+2}*Z{6*i+3}'), (-1, f'X{6*i+2}*X{6*i+3}'), (-2, f'Z{6*i+4}*Z{6*i+5}')]
        Hpen_block+= [(-int(2**(m-1)), f'Y{6*i}*Y{6*i+1}'), (int(2**(m-1)), f"Y{6*(i+1)-2}*Y{6*(i+1)-1}")]
    Hpen = blocks2Mat(blockSize * blockNum, Hpen_block)
    return Hpen
blockSize = 6
blockNum = 2
H = getHpen(blockNum)
n = blockSize * blockNum
print(H.shape)
config = {"target": 0, "distance": 2, "depth": 2, "thres": 1}
# P = get_zero_projector(H, 64)
P, HpenInverse = zero_eigen_projector(H)
print(P.shape)
# # Hinv = np.linalg.pinv(H)
# space = getSpace(n, H, config)

# %%
import pennylane as qml

def load_or_precompute(n_block):
    gx, gz = 1, 3
    if not os.path.exists(f"Hpen_{n_block}blocks.npy"):
        n_phys = 4 * n_block
        Hpen_terms = []
        for i in range(n_block):
            Hpen_terms += [(gx, f'X{4*i}*X{4*i+1}'), (gz, f'Z{4*i}*Z{4*i+1}'),
                        (gx, f'X{4*i+2}*X{4*i+3}'), (gz, f'Z{4*i+2}*Z{4*i+3}')]
        Hpen = blocks2Mat(n_phys, Hpen_terms)
        P0, HpenInverse = get_P0_HpenInverse(Hpen)
        Uenc = getU(4, n_block)
        Penc = Uenc @ Uenc.conj().T
        np.save(f'Hpen_{n_block}blocks', Hpen)
        np.save(f'HpenInverse_{n_block}blocks', HpenInverse)
        np.save(f'P0_{n_block}blocks', P0)
        np.save(f'Uenc_{n_block}blocks', Uenc)
        np.save(f'Penc_{n_block}blocks', Penc)
    
    Hpen = np.load(f'Hpen_{n_block}blocks.npy')
    HpenInverse = np.load(f'HpenInverse_{n_block}blocks.npy')
    P0 = np.load(f'P0_{n_block}blocks.npy')
    Uenc = np.load(f'Uenc_{n_block}blocks.npy')
    Penc = np.load(f'Penc_{n_block}blocks.npy')
    
    return Hpen, HpenInverse, P0, Uenc, Penc

def test_leakage_and_get_logical_interaction(HpenInverse, P0, Penc, Uenc, Henc):
    A = Henc @ HpenInverse @ Henc
    off_diag = (P0 - Penc) @ A @ Penc
    if checkSame(off_diag, np.zeros(off_diag.shape)):
        print("no leakage")
    else:
        print("yes leakage")
    Hlogi = - Uenc.conj().T @ A @ Uenc
    return qml.pauli_decompose(Hlogi)
    

def get_P0_HpenInverse(Hpen):
    """
    Make sure that the eigenvalues of Hpen are all integers
    """
    e, u = np.linalg.eigh(Hpen)

    a = np.array([1 if abs(x) < 0.5 else 0 for x in e])
    P0 = u @ (np.expand_dims(a, axis=1) * u.conj().T)

    b = np.array([1/x if abs(x) > 0.5 else 0 for x in e])
    HpenInverse = u @ (np.expand_dims(b, axis=1) * u.conj().T)

    return P0, HpenInverse

qml.pauli_decompose(np.array(X))

# %%
a1 = [("00", "11"), ("00", "11"), ("00", "11")]
a2 = [("00", "11"), ("01", "10"), ("01", "-10")]
a3 = [("00", "-11"), ("00", "-11"), ("00", "-11")]
a4 = [("00", "-11"), ("01", "-10"), ("01", "10")]
a5 = [("01", "10"), ("00", "11"), ("00", "-11")]
a6 = [("01", "10"), ("01", "10"), ("01", "10")]
a7 = [("01", "-10"), ("00", "-11"), ("00", "11")]
a8 = [("01", "-10"), ("01", "-10"), ("01", "-10")]
ap = [a5, a6, a1, a2, a3, a4, a7, a8]
eff = [1, 1j, 1, 1j, 1j, -1, -1j, 1]


def vec(a):
    v1, v2, v3 = ket2Vec(2, a[0]), ket2Vec(2, a[1]), ket2Vec(2, a[2])
    return np.kron(np.kron(v1, v2), v3)

Hsingle = getHpen(1)
vecs = []
for i in range(len(ap)):
    a = ap[i]
    v = eff[i] * vec(a)
    # print(LA.norm(v))
    result = Hsingle @ v - np.zeros(v.shape)
    vecs.append(v)
    # print(LA.norm(result) < 1e-4)
U = np.column_stack(tuple(vecs))

# %%
H = getHpen(blockNum)
HpenInverse2 = np.linalg.pinv(H)

# %%
HI =  H @ HpenInverse2 @ H
Uenc = np.kron(U, U)
Penc = Uenc @ Uenc.conj().T
print(np.allclose(HI, H))
print(f"HpenInverse is hermitian? {np.allclose(HpenInverse2, HpenInverse2.conj().T)}")

# %%
import itertools

def generate_henc_terms():
    """
    Generates a list of all possible Henc terms, where each term is a
    list of two unique (1, 'P_string') tuples.
    """
    
    # --- Step 1: Generate all possible single (1, 'P_string') tuples ---
    
    all_single_terms = []
    paulis = ['X', 'Y', 'Z']
    indices_low = range(6)      # 0 to 5
    indices_high = range(6, 12) # 6 to 11

    # Loop through all combinations to create the canonical single terms
    for p1 in paulis:
        for i1 in indices_low:
            for p2 in paulis:
                for i2 in indices_high:
                    # Create the canonical pauli string, e.g., 'X1*Y6'
                    # The low index (0-5) is always first.
                    pauli_string = f'{p1}{i1}*{p2}{i2}'
                    
                    # Create the tuple as defined
                    single_term = (1, pauli_string)
                    all_single_terms.append(single_term)
    
    num_single_terms = len(all_single_terms)
    # This should be (3 paulis * 6 indices) * (3 paulis * 6 indices) = 18 * 18 = 324
    print(f"Generated {num_single_terms} unique single terms.")
    
    
    # --- Step 2: Generate all unique pairs of these single terms ---
    
    # Use itertools.combinations(iterable, 2) to get all unique pairs.
    # This automatically handles the constraint that [(A, B)] is the 
    # same as [(B, A)] by only generating one of them.
    # It produces tuples of tuples, e.g., ((1, 'A'), (1, 'B'))
    
    final_henc_terms = []
    for term_pair_tuple in itertools.combinations(all_single_terms, 2):
        # Convert the tuple of tuples to the desired list of tuples
        final_henc_terms.append(list(term_pair_tuple))
            
    return final_henc_terms
all_terms = generate_henc_terms()
    
# The total number of pairs is "n C 2" or (n * (n-1)) / 2
# (324 * 323) / 2 = 52,326
print(f"\nGenerated a total of {len(all_terms)} unique Henc_terms.")

print("\n--- Sample of the first 5 terms ---")
for i in range(5):
    print(all_terms[i])
    
print("\n--- Sample of the last 5 terms ---")
for i in range(1, 6):
    print(all_terms[-i])

# %%
with open('result', 'w') as f:
    for i in range(303,304):

        Henc_terms = all_terms[i]

        Henc = blocks2Mat(12, Henc_terms)
        # print(f"Henc is hermitian? {np.allclose(Henc, Henc.conj().T)}")
        # print(f"Uenc is hermitian? {np.allclose(Uenc, Uenc.conj().T)}")
        print(i)
        res = test_leakage_and_get_logical_interaction(HpenInverse2, P, Penc, Uenc, Henc)
        f.write(f"{i}: {Henc_terms}\n {res}\n\n")

# %%



