import numpy as np

from .base_quantum import BaseQuantum
from .qobj import Qobj
from .channel import Channel
from .routines import _SIGMA_I, _SIGMA_X, _SIGMA_Y, _SIGMA_Z


class Gate(BaseQuantum):
    """Class for representing quantum gates

    Parameters
    ----------
    data : array-like
        Matrix representation of a quantum gate

    Attributes
    ----------
    dim : int
        Number of qubits
    H : Gate (property)
        Adjoint matrix of the quantum gate
    matrix : numpy 2-D array (property)
        Matrix representation of the quantum gate
    T : Gate (property)
        Transpose of the quantum gate

    Methods
    -------
    as_channel()
        Convert Gate to Channel
    conj()
        Conjugate of the quantum gate
    copy()
        Create a copy of this Gate instance
    kron()
        Kronecker product of 2 Gate instances
    trace()
        Trace of the quantum gate
    transform()
        Apply this gate to a quantum state
    """
    def __init__(self, data):
        self._matrix = np.array(data, dtype=np.complex128)
        self.dim = int(np.log2(self._matrix.shape[0]))

    @property
    def matrix(self):
        """Quantum gate in a matrix form"""
        return self._matrix

    @matrix.setter
    def matrix(self, data):
        self._matrix = np.array(data, dtype=np.complex128)
        self.dim = int(np.log2(self._matrix.shape[0]))

    def transform(self, state):
        """Apply this gate to the state: U @ rho @ U.H"""
        return Qobj((self @ state @ self.H).matrix)

    def as_channel(self):
        """Return a channel representation of this gate"""
        return Channel(self.transform, self.dim)

    def __repr__(self):
        return 'Quantum gate\n' + repr(self.matrix)


# One-qubit gates

def PHASE(theta):
    return Gate([
        [1, 0],
        [0, np.exp(1j * theta)],
    ])


def RX(theta):
    return Gate([
        [np.cos(theta/2), -1j * np.sin(theta/2)],
        [-1j * np.sin(theta/2), np.cos(theta/2)],
    ])


def RY(theta):
    return Gate([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)],
    ])


def RZ(theta):
    return Gate([
        [np.exp(-theta/2), 0],
        [0, np.exp(theta/2)],
    ])


Id = Gate(_SIGMA_I)
X = Gate(_SIGMA_X)
Y = Gate(_SIGMA_Y)
Z = Gate(_SIGMA_Z)
H = Gate([
    [1, 1],
    [1, -1],
]) / np.sqrt(2)
T = PHASE(np.pi/4)
S = PHASE(np.pi/2)

# Two-qubit gates

CNOT = Gate([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
])

CY = Gate([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, -1j],
    [0, 0, 1j, 0],
])

CZ = Gate([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1],
])

SWAP = Gate([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
])

ISWAP = Gate([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1],
])
