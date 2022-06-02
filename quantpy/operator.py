from copy import deepcopy

import numpy as np

from .base_quantum import BaseQuantum
from .qobj import Qobj
from .routines import _SIGMA_I, _SIGMA_X, _SIGMA_Y, _SIGMA_Z, _vec2mat


class Operator(BaseQuantum):
    """Class for representing quantum operators

    Parameters
    ----------
    data : array-like
        Matrix representation of a quantum operator

    Attributes
    ----------
    H : Operator (property)
        Adjoint matrix of the quantum operator
    matrix : numpy 2-D array (property)
        Matrix representation of the quantum operator
    n_qubits : int
        Number of qubits
    T : Operator (property)
        Transpose of the quantum operator

    Methods
    -------
    as_channel()
        Convert Operator to Channel
    conj()
        Conjugate of the quantum operator
    copy()
        Create a copy of this Operator instance
    kron()
        Kronecker product of 2 Operator instances
    trace()
        Trace of the quantum operator
    transform()
        Apply this operator to a quantum state
    """

    def __init__(self, data):
        if isinstance(data, self.__class__):
            self.__dict__ = deepcopy(data.__dict__)
        self._matrix = np.array(data, dtype=np.complex128)
        self.n_qubits = int(np.log2(self._matrix.shape[0]))

    @property
    def matrix(self):
        """Quantum Operator in a matrix form"""
        return self._matrix

    @matrix.setter
    def matrix(self, data):
        self._matrix = np.array(data, dtype=np.complex128)
        self.n_qubits = int(np.log2(self._matrix.shape[0]))

    def transform(self, state):
        """Apply this Operator to the state: U @ rho @ U.H"""
        return Qobj((self @ state @ self.H).matrix)

    def as_channel(self):
        """Return a channel representation of this Operator"""
        from .channel import Channel

        return Channel(self.transform, self.n_qubits)

    def trace(self):
        """Trace of the quantum object"""
        return np.trace(self.matrix)

    def __repr__(self):
        return "Quantum Operator\n" + repr(self.matrix)


# One-qubit gates

# noinspection PyPep8Naming
def PHASE(theta):
    return Operator(
        [
            [1, 0],
            [0, np.exp(1j * theta)],
        ]
    )


# noinspection PyPep8Naming
def RX(theta):
    return Operator(
        [
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )


# noinspection PyPep8Naming
def RY(theta):
    return Operator(
        [
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )


# noinspection PyPep8Naming
def RZ(theta):
    return Operator(
        [
            [np.exp(-0.5j * theta), 0],
            [0, np.exp(0.5j * theta)],
        ]
    )


Id = Operator(_SIGMA_I)
X = Operator(_SIGMA_X)
Y = Operator(_SIGMA_Y)
Z = Operator(_SIGMA_Z)
H = Operator(
    [
        [1, 1],
        [1, -1],
    ]
) / np.sqrt(2)
T = PHASE(np.pi / 4)
S = PHASE(np.pi / 2)

# Two-qubit gates

CNOT = Operator(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]
)

CY = Operator(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0],
    ]
)

CZ = Operator(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ]
)

SWAP = Operator(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
)

ISWAP = Operator(
    [
        [1, 0, 0, 0],
        [0, 0, 1j, 0],
        [0, 1j, 0, 0],
        [0, 0, 0, 1],
    ]
)

MS = Operator(
    [
        [1, 0, 0, 1j],
        [0, 1, -1j, 0],
        [0, -1j, 1, 0],
        [1j, 0, 0, 1],
    ]
) / np.sqrt(2)


Toffoli = Operator(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ]
)


Fredkin = Operator(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
)


def _choi_to_kraus(choi):
    EPS = 1e-15
    eigvals, eigvecs = choi.eig()
    eigvecs = list(eigvecs.T)
    return [Operator(_vec2mat(vec) * np.sqrt(val)) for val, vec in zip(eigvals, eigvecs) if abs(val) > EPS]
