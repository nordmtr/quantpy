import numpy as np

from .base_quantum import BaseQuantum
from .qobj import Qobj


class Gate(BaseQuantum):
    def __init__(self, data):
        self._matrix = data

    @property
    def matrix(self):
        """Quantum object in a matrix form"""
        return self._matrix

    @matrix.setter
    def matrix(self, data):
        self._matrix = np.array(data)

    def transform(self, rho):
        return Qobj((self @ rho @ self.H).matrix)

    def __repr__(self):
        return 'Quantum gate\n' + repr(self.matrix)
