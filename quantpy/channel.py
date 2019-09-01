import numpy as np

from copy import deepcopy

from .base_quantum import BaseQuantum
from .operator import _choi_to_kraus, Z, Operator
from .routines import generate_single_entries, kron
from .qobj import Qobj, fully_mixed


class Channel(BaseQuantum):
    """Class for representing quantum gates

    Parameters
    ----------
    data : callable, numpy 2-D array, Qobj or list
        If callable, treated as a transformation function. `n_qubits` argument is necessary in this case.
            Note: using non-linear functions can lead to unpredictable results
        If numpy 2-D array or Qobj, treated as a Choi matrix
        If list, treated as Kraus representation
    n_qubits : int or None, default=None (optional)
        Number of qubits

    Attributes
    ----------
    choi : Qobj (property)
        Choi matrix of the channel
    H : Channel (property)
        Channel with adjoint Choi matrix
    n_qubits : int
        Number of qubits
    T : Channel (property)
        Channel with transposed Choi matrix

    Methods
    -------
    conj()
        Channel with conjugated Choi matrix
    copy()
        Create a copy of this Gate instance
    kraus()
        Kraus representation of the channel
    kron()
        Kronecker product of 2 Qobj instances
    set_func()
        Set a new channel via function
    trace()
        Trace of the quantum object
    transform()
        Apply this channel to a quantum state
    """
    def __init__(self, data, n_qubits=None):
        self._types = set()
        if isinstance(data, self.__class__):
            self.__dict__ = deepcopy(data.__dict__)
        elif callable(data):
            self._choi = None
            self._kraus = None
            self._func = data
            self._types.add('func')
            if n_qubits is None:
                raise ValueError('`n_qubits` argument is compulsory when using init with function')
            self.n_qubits = n_qubits
        elif isinstance(data, np.ndarray) or isinstance(data, Qobj):
            self._choi = Qobj(data)
            self._func = None
            self._kraus = None
            self._types.add('choi')
            self.n_qubits = self._choi.n_qubits / 2
        elif isinstance(data, list):
            self._choi = None
            self._func = None
            self._kraus = data
            self._types.add('kraus')
            self.n_qubits = data[0].n_qubits
        else:
            raise ValueError('Invalid data format')

    def set_func(self, data, n_qubits):
        """Set a new transformation function that defines the channel"""
        self._types.discard('choi')
        self._types.discard('kraus')
        self._func = data
        self.n_qubits = n_qubits
        self._types.add('func')

    @property
    def choi(self):
        """Choi matrix of the channel"""
        if 'choi' not in self._types:
            self._choi = Qobj(np.zeros((4 ** self.n_qubits, 4 ** self.n_qubits), dtype=np.complex128))
            for single_entry in generate_single_entries(2 ** self.n_qubits):
                self._choi += kron(Qobj(single_entry), self.transform(single_entry))
            self._types.add('choi')
        return self._choi

    @choi.setter
    def choi(self, data):
        self._types.discard('func')
        self._types.discard('kraus')
        if not isinstance(data, Qobj):
            data = Qobj(data)
        self._choi = data
        self.n_qubits = int(np.log2(data.shape[0]) / 2)
        self._types.add('choi')

    @property
    def kraus(self):
        if 'kraus' not in self._types:
            self._kraus = _choi_to_kraus(self.choi)
            self._types.add('kraus')
        return self._kraus

    def transform(self, state):
        """Apply this channel to the quantum state"""
        if not isinstance(state, Qobj):
            state = Qobj(state)
        if 'kraus' in self._types:
            output_state = np.sum([oper.transform(state) for oper in self.kraus])
        elif 'func' in self._types:
            output_state = self._func(state)
        else:  # compute output state using Choi matrix
            common_state = kron(state.T, Qobj(np.eye(2 ** self.n_qubits)))
            output_state = (common_state @ self.choi).ptrace(list(range(self.n_qubits, 2 * self.n_qubits)))
        return output_state

    @property
    def T(self):
        """Transpose of the quantum object"""
        return self.__class__(self.choi.T)

    @property
    def H(self):
        """Adjoint matrix of the quantum object"""
        return self.__class__(self.choi.H)

    def conj(self):
        """Conjugate of the quantum object"""
        return self.__class__(self.choi.conj())

    def __repr__(self):
        return 'Quantum channel w Choi matrix\n' + repr(self.choi.matrix)


def depolarizing(p=1, n_qubits=1):
    """Depolarizing channel with probability `p`
    rho -> p * Id / (2^n_qubits) + (1-p) * rho
    """
    return Channel(lambda rho: p * rho.trace() * fully_mixed(n_qubits) + (1 - p) * rho, n_qubits)


def dephasing(p=1, n_qubits=1):
    """Dephasing channel with probability `p`
    rho -> (1-p) * rho + p * Z @ rho @ Z
    """
    return Channel(lambda rho: p * Z.transform(rho) + (1-p) * rho, n_qubits)


def amplitude_damping(gamma):
    """Amplitude damping channel with probability of decay `gamma`"""
    kraus_list = [
        np.sqrt(gamma) * Operator([[0, 1], [0, 0]]),
        Operator([[1, 0], [0, 0]]) + np.sqrt(1-gamma) * Operator([[0, 0], [0, 1]]),
    ]
    return Channel(kraus_list)
