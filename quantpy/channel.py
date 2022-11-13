import sys
from copy import deepcopy

import numpy as np

from .base_quantum import BaseQuantum
from .operator import H, Operator, Z, _choi_to_kraus
from .qobj import Qobj, fully_mixed
from .routines import generate_single_entries, kron


class Channel(BaseQuantum):
    """Class for representing quantum gates

    Parameters
    ----------
    data : callable, numpy 2-D array, Qobj or list
        If callable, treated as a transformation function.
        `n_qubits` argument is necessary in this case.
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
    is_cptp()
        Check if channel is completely positive and trace-preserving
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
            self._types.add("func")
            if n_qubits is None:
                raise ValueError("`n_qubits` argument is compulsory when using init with function")
            self.n_qubits = n_qubits
        elif isinstance(data, np.ndarray) or isinstance(data, Qobj):
            self._choi = Qobj(data)
            self._func = None
            self._kraus = None
            self._types.add("choi")
            self.n_qubits = int(self._choi.n_qubits / 2)
        elif isinstance(data, list):
            self._choi = None
            self._func = None
            self._kraus = data
            self._types.add("kraus")
            self.n_qubits = data[0].n_qubits
        else:
            raise ValueError("Invalid data format")

    def set_func(self, data, n_qubits):
        """Set a new transformation function that defines the channel"""
        self._types.discard("choi")
        self._types.discard("kraus")
        self._func = data
        self.n_qubits = n_qubits
        self._types.add("func")

    @property
    def choi(self):
        """Choi matrix of the channel"""
        if "choi" not in self._types:
            self._choi = Qobj(np.zeros((4**self.n_qubits, 4**self.n_qubits), dtype=np.complex128))
            for single_entry in generate_single_entries(2**self.n_qubits):
                self._choi += kron(Qobj(single_entry), self.transform(single_entry))
            self._types.add("choi")
        return self._choi

    @choi.setter
    def choi(self, data):
        self._types.discard("func")
        self._types.discard("kraus")
        if not isinstance(data, Qobj):
            data = Qobj(data)
        elif not isinstance(data, np.ndarray):
            raise ValueError("Invalid data format")
        self._choi = data
        self.n_qubits = int(np.log2(data.shape[0]) / 2)
        self._types.add("choi")

    @property
    def kraus(self):
        """Kraus representation of the channel"""
        if "kraus" not in self._types:
            self._kraus = _choi_to_kraus(self.choi)
            self._types.add("kraus")
        return self._kraus

    @kraus.setter
    def kraus(self, data):
        self._types.discard("func")
        self._types.discard("choi")
        if not isinstance(data, list):
            raise ValueError("Invalid data format")
        self._kraus = data
        self.n_qubits = data[0].n_qubits

    def transform(self, state):
        """Apply this channel to the quantum state"""
        if not isinstance(state, Qobj):
            state = Qobj(state)
        if "kraus" in self._types:
            output_state = np.sum([oper.transform(state) for oper in self.kraus])
        elif "func" in self._types:
            output_state = self._func(state)
        else:  # compute output state using Choi matrix
            common_state = kron(state.T, Qobj(np.eye(2**self.n_qubits)))
            output_state = (common_state @ self.choi).ptrace(list(range(self.n_qubits, 2 * self.n_qubits)))
        return output_state

    def is_cptp(self, atol=1e-5, verbose=True):
        """Check if channel is trace-preserving and completely positive
        `atol` param sets absolute tolerance level for the comparison.
        See :ref:`numpy.allclose` for detailed documentation."""
        rho_in = self.choi.ptrace(list(range(self.n_qubits)))
        tp_flag = np.allclose(rho_in.matrix, np.eye(2**rho_in.n_qubits), atol=atol)
        cp_flag = np.allclose(np.minimum(np.real(self.choi.eig()[0]), 0), 0, atol=atol)
        if tp_flag and cp_flag:
            return True
        if not tp_flag and verbose:
            print("Not trace-preserving", file=sys.stderr)
        if not cp_flag and verbose:
            print("Not completely positive", file=sys.stderr)
        return False

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
        return "Quantum channel with Choi matrix\n" + repr(self.choi.matrix)

    def _repr_latex_(self):
        return r"Choi matrix: " + Qobj(self.choi.matrix)._repr_latex_()

    def __eq__(self, other):
        return np.array_equal(self.choi.matrix, other.choi.matrix)

    def __ne__(self, other):
        return not np.array_equal(self.choi.matrix, other.choi.matrix)

    def __neg__(self):
        return self.__class__(-self.choi)

    def __add__(self, other):
        return self.__class__(self.choi + other.choi)

    def __sub__(self, other):
        return self.__class__(self.choi - other.choi)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__class__(self.choi * other)
        else:
            raise ValueError("Only multiplication by a scalar is allowed")

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__class__(self.choi / other)
        else:
            raise ValueError("Only division by a scalar is allowed")

    def __iadd__(self, other):
        self.choi = self.choi + other.choi
        return self

    def __isub__(self, other):
        self.choi = self.choi - other.choi
        return self

    def __imul__(self, other):
        if type(other) in (int, float, complex):
            self.choi = self.choi * other
            return self
        else:
            raise ValueError("Only multiplication by a scalar is supported")

    def __idiv__(self, other):
        if type(other) in (int, float, complex):
            self.choi = self.choi / other
            return self
        else:
            raise ValueError("Only division by a scalar is supported")

    def __rmul__(self, other):
        return self.__mul__(other)


def depolarizing(p=1, n_qubits=1):
    """Depolarizing channel with probability `p`
    rho -> p * Id / (dim) + (1-p) * rho
    """
    return Channel(lambda rho: p * rho.trace() * fully_mixed(n_qubits) + (1 - p) * rho, n_qubits)


def dephasing(p=1, n_qubits=1):
    """Dephasing channel with probability `p`
    rho -> (1-p) * rho + p * Z @ rho @ Z
    """
    return Channel(lambda rho: p * Z.transform(rho) + (1 - p) * rho, n_qubits)


def amplitude_damping(gamma):
    """Amplitude damping channel with probability of decay `gamma`"""
    kraus_list = [
        np.sqrt(gamma) * Operator([[0, 1], [0, 0]]),
        Operator([[1, 0], [0, 0]]) + np.sqrt(1 - gamma) * Operator([[0, 0], [0, 1]]),
    ]
    return Channel(kraus_list)


def walsh_hadamard(n_qubits):
    operator = H
    for _ in range(n_qubits - 1):
        operator = operator.kron(H)
    return operator.as_channel()


def depolarize(channel, p):
    """Add p-depolarization to channel"""
    return Channel(
        lambda rho: (1 - p) * channel.transform(rho) + p * rho.trace() * fully_mixed(channel.n_qubits), channel.n_qubits
    )
