import numpy as np

from .base_quantum import BaseQuantum
from .routines import generate_single_entries
from .qobj import Qobj
from .routines import kron


class Channel(BaseQuantum):
    """Class for representing quantum gates

    Parameters
    ----------
    data : callable, array-like or Qobj
        If callable, treated as a transformation function. `dim` argument is necessary in this case.
        If array-like or Qobj, treated as a Choi matrix
    dim : int or None, default=None (optional)
        Number of qubits

    Attributes
    ----------
    choi : Qobj (property)
        Choi matrix of the channel
    dim : int
        Number of qubits
    H : Gate (property)
        Adjoint matrix of the quantum object
    T : Gate (property)
        Transpose of the quantum object

    Methods
    -------
    conj()
        Conjugate of the quantum object
    copy()
        Create a copy of this Gate instance
    kron()
        Kronecker product of 2 Qobj instances
    set_func()
        Set a new channel via function
    trace()
        Trace of the quantum object
    transform()
        Apply this channel to a quantum state
    """
    def __init__(self, data, dim=None):
        self._types = set()
        if callable(data):
            self._choi = None
            self._func = data
            self._types.add('func')
            if dim is None:
                raise ValueError('`dim` argument is compulsory when using init with function')
            self.dim = dim
        elif isinstance(data, np.ndarray) or isinstance(data, Qobj):
            self._choi = Qobj(data)
            self._func = None
            self._types.add('choi')
            self.dim = int(np.log2(data.shape[0]) / 2)
        else:
            raise ValueError('Invalid data format')

    def set_func(self, data, dim):
        """Set a new transformation function that defines the channel"""
        self._types.add('func')
        self._types.discard('choi')
        self._func = data
        self.dim = dim

    @property
    def choi(self):
        """Choi matrix of the channel"""
        if 'choi' not in self._types:
            self._types.add('choi')
            self._choi = Qobj(np.zeros((4 ** self.dim, 4 ** self.dim), dtype=np.complex128))
            for single_entry in generate_single_entries(2 ** self.dim):
                self._choi += kron(Qobj(single_entry), self.transform(single_entry))
        return self._choi

    @choi.setter
    def choi(self, data):
        self._types.add('choi')
        self._types.discard('func')
        if not isinstance(data, Qobj):
            data = Qobj(data)
        self._choi = data
        self.dim = int(np.log2(data.shape[0]) / 2)

    def transform(self, state):
        """Apply this channel to the quantum state"""
        if not isinstance(state, Qobj):
            state = Qobj(state)
        if 'func' in self._types:
            output_state = self._func(state)
        else:  # compute output state using Choi matrix
            common_state = kron(state.T, Qobj(np.eye(2 ** self.dim)))
            output_state = (common_state @ self.choi).ptrace(list(range(self.dim, 2 * self.dim)))
        return output_state

    def __repr__(self):
        return 'Quantum channel w Choi matrix\n' + repr(self.choi.matrix)
