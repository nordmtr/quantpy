import numpy as np
import scipy.linalg as la
import sys


def hs(A, B):
    """ Hilbert-Schmidt distance between two matrices """
    dist = np.sqrt(abs(np.trace((A - B) * (A - B)))) / np.sqrt(2)
    if dist < 1e-15:
        return 0
    else:
        return dist


def trace(A, B):
    """ Trace distance between two matrices """
    dist = abs(np.trace(la.sqrtm((A - B) @ (A - B)))) / 2
    if dist < 1e-15:
        return 0
    else:
        return dist


def infidelity(A, B):
    """ Infidelity between two matrices """
    dist = 1 - np.abs(np.trace(la.sqrtm(la.sqrtm(A) @ B @ la.sqrtm(A))) ** 2)
    if dist < 1e-15:
        return 0
    else:
        return dist


def product(A, B):
    """ Hermitian inner product in matrix space """
    return np.trace(A @ np.conj(B.T), dtype=np.complex128)


def density(psi):
    """
    Construct a density matrix of a pure state
    Input:
        psi = [x1, x2]
    """
    return np.array(np.matrix(psi, dtype=np.complex128).T @ np.conj(np.matrix(psi, dtype=np.complex128)))


def left_inv(A):
    return la.inv(A.T @ A) @ A.T


# Pauli basis

SIGMA_I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

PAULI_1 = [SIGMA_I, SIGMA_X, SIGMA_Y, SIGMA_Z]


def generate_pauli(dim):
    basis = PAULI_1
    for _ in range(dim - 1):
        basis = np.kron(basis, PAULI_1)
    return basis


class Qstate:
    ''' Basic class for quantum states '''
    def __init__(self, data):
        if isinstance(data, list):
            data = np.array(data)
        if len(data.shape) == 1:
            self._matrix = None
            self._bloch = data
            self._type = 'bloch'
            self.dim = int(np.log2(data.shape[0]) / 2)
        elif len(data.shape) == 2:
            self._matrix = data
            self._bloch = None
            self._type = 'matrix'
            self.dim = int(np.log2(data.shape[0]))
        else:
            raise ValueError('Invalid data format')

    @property
    def matrix(self):
        if self._type == 'matrix':
            return self._matrix
        else:
            basis = generate_pauli(self.dim)
            matrix = np.matrix(np.zeros((2 ** self.dim, 2 ** self.dim)), dtype=np.complex128)
            for i in range(4 ** self.dim):
                matrix += basis[i] * self._bloch[i]
            return matrix / (2 ** self.dim)

    @matrix.setter
    def matrix(self, data):
        self._type = 'matrix'
        self._matrix = data

    @property
    def bloch(self):
        if self._type == 'bloch':
            return self._bloch
        else:
            basis = generate_pauli(self.dim)
            return np.array([np.real(product(basis_element, self._matrix)) for basis_element in basis])

    @bloch.setter
    def bloch(self, data):
        if isinstance(data, list):
            data = np.array(data)
        self._type = 'bloch'
        self._bloch = data
