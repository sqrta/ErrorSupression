import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from Hamil_search import *
from simulation import blocks2Mat


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


if __name__ == '__main__':
    # ---
    # ## Example Usage
    # ---

    blockSize = 6
    blockNum = 2
    H = getHpen(blockNum)
    is_hermitian = np.allclose(H, H.conj().T)
    N = 100
    rank = 97
    k_null = N - rank

    # Create two random sparse matrices
    A = sp.random(N, rank, density=0.5, format='csc', dtype=np.complex128)
    B = sp.random(rank, N, density=0.5, format='csc', dtype=np.complex128)
    # H = A @ B

    print(f"H is hermitian? {is_hermitian}, shape: {H.shape}")
    config = {"target": 0, "distance": 2, "depth": 2, "thres": 1}
    P = zero_eigen_projector(H)

    print(f"\nComputed {P.shape} dense projector matrix P.")

    # 3. Verification
    if P.shape[0] > 0:
        # Test 1: P should be Hermitian (P == P.conj().T)
        is_hermitian = np.allclose(P, P.conj().T)
        print(f"Verification: Is P Hermitian? {is_hermitian}")

        # Test 2: P should be idempotent (P @ P == P)
        is_idempotent = np.allclose(P @ P, P)
        print(f"Verification: Is P idempotent (P*P = P)? {is_idempotent}")
        
        # Test 3: P must project onto the null space, so H @ P should be all zeros.
        # We compute the norm to check if it's close to 0.
        H_P_norm = np.linalg.norm((H@P))
        print(f"Verification: Norm of (H @ P) = {H_P_norm} (should be near 0)")