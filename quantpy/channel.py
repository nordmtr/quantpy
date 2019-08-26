import numpy as np

from .base_quantum import BaseQuantum
from .routines import generate_single_entries
from .qobj import Qobj


class Channel(BaseQuantum):
    def __init__(self, data, dim=None):
        self._types = set()
        if callable(data):
            self._choi = None
            self._func = data
            self._types.add('func')
            if dim is None:
                raise ValueError('`dim` argument is compulsory when using init with function')
            self.dim = dim
        elif isinstance(data, np.ndarray):
            self._choi = data
            self._func = None
            self._types.add('choi')
            self.dim = int(np.log2(data.shape[0]) / 2)
        else:
            raise ValueError('Invalid data format')

    # @property
    # def func(self):
    #     if 'func' not in self._types:
    #         self._types.add('func')  # TODO: derive transformation using Choi matrix
    #     return self._func

    def set_func(self, data, dim):
        self._types.add('func')
        self._types.discard('choi')
        self._func = data
        self.dim = dim

    @property
    def choi(self):
        if 'choi' not in self._types:
            self._types.add('choi')
            self._choi = np.array((4 ** self.dim, 4 ** self.dim), dtype=np.complex128)
            for single_entry in generate_single_entries(4 ** self.dim):
                self._choi += np.kron(single_entry, self.transform(single_entry))
        return self._choi

    @choi.setter
    def choi(self, data):
        self._types.add('choi')
        self._types.discard('func')
        self._choi = data
        self.dim = int(np.log2(data.shape[0]) / 2)

    def transform(self, state):
        common_state = state.T.tensordot(Qobj(np.eye(2 ** self.dim)))
        return common_state.ptrace(list(range(self.dim, 2 * self.dim)))
