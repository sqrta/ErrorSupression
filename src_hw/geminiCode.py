import numpy as np
from scipy.sparse import csc_matrix, random
import scipy.sparse.linalg as sps_linalg
import scipy.sparse as sps
import warnings
from Hamil_search1 import *
from simulation import blocks2Mat

def get_zero_projector(H, k_expected_null_dim=10, tol=1e-9):
    """
    Computes the dense projector matrix onto the null space (zero eigenspace)
    of a sparse matrix H.

    The projector P is defined as sum(v @ v.conj().T) for all v in the
    orthonormal basis of the null space. This is equivalent to V_null @ V_null.conj().T
    where V_null's columns are the orthonormal basis vectors.

    Args:
        H (scipy.sparse.spmatrix): The input sparse matrix (N x N).
        k_expected_null_dim (int): Your *estimate* of the null space dimension.
                                    This value MUST be >= the true dimension
                                    for this method to work.
        tol (float): The numerical tolerance to consider a singular value
                     as zero.

    Returns:
        numpy.ndarray: The (N x N) dense projector matrix P.
    """
    N, M = H.shape
    if N != M:
        raise ValueError(f"Matrix must be square, but got shape {H.shape}")

    if k_expected_null_dim >= N - 1:
        warnings.warn(f"k ({k_expected_null_dim}) is too large for eigs/svds. "
                      f"Reducing to {N - 2}.")
        k_expected_null_dim = N - 2

    print(f"Computing SVD for {k_expected_null_dim} smallest singular values...")
    try:
        # 'svds' finds the k *largest* singular values by default.
        # 'which='SM'' tells it to find the ones with the *Smallest Magnitude*.
        # 'u' = Left singular vectors
        # 's' = Singular values (sorted smallest to largest)
        # 'vh' = *Conjugate Transpose* of Right singular vectors
        u, s, vh = sps_linalg.svds(H, k=k_expected_null_dim, which='SM')

    except RuntimeError as e:
        print(f"SVD computation did not converge: {e}")
        print("Try increasing 'k_expected_null_dim' or checking your matrix.")
        return np.zeros((N, N), dtype=H.dtype)
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.zeros((N, N), dtype=H.dtype)

    # Find which singular values are effectively zero
    zero_mask = s < tol
    num_found = np.sum(zero_mask)

    if num_found == 0:
        print("No zero singular values found. Null space is empty.")
        return np.zeros((N, N), dtype=H.dtype)

    if num_found == k_expected_null_dim:
        warnings.warn(f"Found {num_found} vectors, which matches k."
                      "The true null space might be larger. "
                      "Rerun with a larger 'k_expected_null_dim'.")

    print(f"Found {num_found} vectors in the null space.")
    
    # The null space is spanned by the *right singular vectors* (the rows of vh)
    # corresponding to the zero singular values.
    # We take the transpose (.T) to get them as columns.
    null_space_basis = vh[zero_mask, :].conj().T
    
    # The basis vectors from svds (columns of V, or rows of vh)
    # are already orthonormal.
    
    # P = V_null @ V_null.conj().T
    projector = null_space_basis @ null_space_basis.conj().T
    
    return projector

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

if __name__ == '__main__':
    # ---
    # ## Example Usage
    # ---

    blockSize = 6
    blockNum = 1
    H = getHpen(blockNum)
    is_hermitian = np.allclose(H, H.conj().T)
    N = 100
    rank = 90
    k_null = N - rank

    # Create two random sparse matrices
    A = sps.random(N, rank, density=0.5, format='csc', dtype=np.complex128)
    B = sps.random(rank, N, density=0.5, format='csc', dtype=np.complex128)
    # H = A @ B

    print(f"H is hermitian? {is_hermitian}, shape: {H}")
    P = get_zero_projector(H, k_expected_null_dim=11, tol=1e-9)

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
        H_P_norm = np.linalg.norm((H @ P))
        print(f"Verification: Norm of (H @ P) = {H_P_norm} (should be near 0)")