import numpy as np
import scipy.linalg as la


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


# def tensordot(A, B):
#     """Return Kronecker product of 2 Qstates. Copy of `Qobj.tensordot`."""
#     return Qobj(A).tensordot(B)
