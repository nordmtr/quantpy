from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np


class BaseQuantum(ABC):
    """Abstract base class for quantum states, gates, operators and channels"""

    @abstractmethod
    def __repr__(self):
        pass

    @property
    def T(self):
        """Transpose of the quantum object"""
        return self.__class__(self.matrix.T)

    @property
    def H(self):
        """Adjoint matrix of the quantum object"""
        return self.__class__(self.matrix.T.conj())

    def conj(self):
        """Conjugate of the quantum object"""
        return self.__class__(self.matrix.conj())

    def copy(self):
        """Create a copy of this instance"""
        return deepcopy(self)

    def kron(self, other):
        """Kronecker product of 2 instances"""
        return self.__class__(np.kron(self.matrix, other.matrix))

    def __eq__(self, other):
        return np.array_equal(self.matrix, other.matrix)

    def __ne__(self, other):
        return not np.array_equal(self.matrix, other.matrix)

    def __neg__(self):
        return self.__class__(-self.matrix)

    def __matmul__(self, other):
        return self.__class__(self.matrix @ other.matrix)

    def __add__(self, other):
        return self.__class__(self.matrix + other.matrix)

    def __sub__(self, other):
        return self.__class__(self.matrix - other.matrix)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__class__(self.matrix * other)
        else:
            raise ValueError("Only multiplication by a scalar is allowed")

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__class__(self.matrix / other)
        else:
            raise ValueError("Only division by a scalar is allowed")

    def __iadd__(self, other):
        self.matrix = self.matrix + other.matrix
        return self

    def __isub__(self, other):
        self.matrix = self.matrix - other.matrix
        return self

    def __imul__(self, other):
        if type(other) in (int, float, complex):
            self.matrix = self.matrix * other
            return self
        else:
            raise ValueError("Only multiplication by a scalar is supported")

    def __idiv__(self, other):
        if type(other) in (int, float, complex):
            self.matrix = self.matrix / other
            return self
        else:
            raise ValueError("Only division by a scalar is supported")

    def __rmul__(self, other):
        return self.__mul__(other)
