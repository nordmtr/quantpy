import numpy as np

from .geometry import product
from .routines import generate_pauli, _density


class Qobj:
    """Basic class for quantum objects."""

    def __init__(self, data=None, is_ket=False):
        self._types = set()  # Set of types which represent the state
        if is_ket:
            data = _density(data)
        data = np.array(data)
        if len(data.shape) == 1:
            self._matrix = None
            self._bloch = data
            self._types.add('bloch')
            self.dim = int(np.log2(data.shape[0]) / 2)
        elif len(data.shape) == 2:
            self._matrix = data
            self._bloch = None
            self._types.add('matrix')
            self.dim = int(np.log2(data.shape[0]))
        else:
            raise ValueError('Invalid data format')

    @property
    def matrix(self):
        if 'matrix' not in self._types:
            self._types.add('matrix')
            basis = generate_pauli(self.dim)
            self._matrix = np.zeros((2 ** self.dim, 2 ** self.dim), dtype=np.complex128)
            for i in range(4 ** self.dim):
                self._matrix += basis[i] * self._bloch[i]
            # self._matrix /= (2 ** self.dim)
        return self._matrix

    @matrix.setter
    def matrix(self, data):
        self._types.add('matrix')
        self._types.discard('bloch')
        self._matrix = data

    @property
    def bloch(self):
        if 'bloch' not in self._types:
            self._types.add('bloch')
            basis = generate_pauli(self.dim)
            self._bloch = np.array(
                [np.real(product(basis_element, self._matrix)) for basis_element in basis]
            ) / (2 ** self.dim)
        return self._bloch

    @bloch.setter
    def bloch(self, data):
        if isinstance(data, list):
            data = np.array(data)
        self._type = 'bloch'
        self._bloch = data

    def tensordot(self, other):
        """Return Kronecker product of 2 Qobj instances."""
        return self.__class__(np.kron(self.matrix, other.matrix))

    def is_pure(self):
        return np.linalg.matrix_rank(self.matrix, tol=1e-10, hermitian=True) == 1

    def __repr__(self):
        return repr(self.matrix)

    def __eq__(self, other):
        return self.matrix == other.matrix

    def __ne__(self, other):
        return self.matrix != other.matrix

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
            raise ValueError('Only multiplication by a scalar is allowed')

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__class__(self.matrix / other)
        else:
            raise ValueError('Only division by a scalar is allowed')

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
            raise ValueError('Only multiplication by a scalar is supported')

    def __idiv__(self, other):
        if type(other) in (int, float, complex):
            self.matrix = self.matrix * other
            return self
        else:
            raise ValueError('Only division by a scalar is supported')

    def __rmul__(self, other):
        return self.__mul__(other)
