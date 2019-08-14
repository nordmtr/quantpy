import numpy as np
import scipy.linalg as la


# Pauli basis

SIGMA_I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

PAULI_1 = [SIGMA_I, SIGMA_X, SIGMA_Y, SIGMA_Z]


def generate_pauli(dim):
    """Generate basis of Pauli matrices for `dim` qubits."""
    basis = PAULI_1
    for _ in range(dim - 1):
        basis = np.kron(basis, PAULI_1)
    return basis


def _density(psi):
    """
    Construct a density matrix of a pure state
    Input:
        psi = [x1, x2]
    """
    return np.array(np.matrix(psi, dtype=np.complex128).T @ np.conj(np.matrix(psi, dtype=np.complex128)))


def _left_inv(A):
    """Return left pseudo-inverse matrix."""
    return la.inv(A.T @ A) @ A.T


def _real_to_complex(z):
    """Real vector of length 2n -> complex of length n"""
    return z[:len(z)//2] + 1j * z[len(z)//2:]


def _complex_to_real(z):
    """Complex vector of length n -> real of length 2n"""
    return np.concatenate((np.real(z), np.imag(z)))


def _matrix_to_real_tril_vec(matrix):
    tril_matrix = la.cholesky(matrix, lower=True)
    complex_tril_vector = tril_matrix[np.tril_indices(tril_matrix.shape[0])]
    return _complex_to_real(complex_tril_vector)


def _real_tril_vec_to_matrix(vector):
    complex_vector = _real_to_complex(vector)
    matrix_shape = int(-0.5 + np.sqrt(2 * len(complex_vector) + 0.25))  # solve a quadratic equation for the shape
    tril_matrix = np.zeros((matrix_shape, matrix_shape), dtype=np.complex128)
    tril_matrix[np.tril_indices(matrix_shape)] = complex_vector
    return tril_matrix @ tril_matrix.T.conj()


# def tensordot(A, B):
#     """Return Kronecker product of 2 Qstates. Copy of `Qobj.tensordot`."""
#     return Qobj(A).tensordot(B)
