import sys
import numpy as np
import scipy.linalg as la

from copy import deepcopy

from .geometry import product
from .routines import generate_pauli, _density


class Qobj:
    """Basic class for representing quantum objects, such as quantum states and operators

    This class supports all simple math operations, as well as a collection of common
    state/operator operations.

    Parameters
    ----------
    data : array-like
        If `data` is 2-D, it is treated as a full matrix
        If `data` is 1-D and `is_ket` is False, it is treated as a bloch vector
        If `data` is 1-D and `is_let` is True, it is treated as a ket vector
    is_ket : bool, default=False

    Attributes
    ----------
    bloch : numpy 1-D array (property)
        A vector, representing the quantum object in Pauli basis (only for Hermitian matrices)
    dim : int
        Number of qubits
    H : Qobj (property)
        Adjoint matrix of the quantum object
    matrix : numpy 2-D array (property)
        Quantum object in a matrix form
    T : Qobj (property)
        Transpose of the quantum object

    Methods
    -------
    copy()
        Create a copy of this Qobj instance
    conj()
        Conjugate of the quantum object
    eig()
        Eigenvalues and eigenvectors of the quantum object
    is_density_matrix()
        Check if the quantum object is valid density matrix
    is_pure()
        Check if the quantum object is rank-1 valid density matrix
    tensordot()
        Kronecker product of 2 Qobj instances
    trace()
        Trace of the quantum object

    Examples
    --------
    >>> qp.Qobj([0.5, 0, 0, 0.5])
    array([[1.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j]])

    >>> qp.Qobj([[1.+0.j, 0.+0.j],
                 [0.+0.j, 0.+0.j]])
    array([[1.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j]])

    >>> qp.Qobj([1, 0], is_ket=True)
    array([[1.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j]])
    """

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
        """Quantum object in a matrix form"""
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
        """A vector, representing the quantum object in Pauli basis"""
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

    def trace(self):
        """Trace of the quantum object"""
        return np.trace(self.matrix)

    def eig(self):
        """Find eigenvalues and eigenvectors of the quantum object

        Returns
        -------
        v : complex numpy 1-D array
            The eigenvalues, each repeated according to its multiplicity
        U : complex numpy 2-D array
            The normalized right eigenvector corresponding to the eigenvalue `v[i]`
            is the column `U[:, i]`

        Raises
        ------
        LinAlgError
            If eigenvalue computation does not converge
        """
        return la.eig(self.matrix)

    def tensordot(self, other):
        """Kronecker product of 2 Qobj instances

        Parameters
        ----------
        other : Qobj

        Returns
        -------
        result : Qobj
            Kronecker product of `self` and `other`
        """
        return self.__class__(np.kron(self.matrix, other.matrix))

    def is_density_matrix(self):
        """Check if the quantum object is a valid density matrix.
        Perform a test for hermiticity, positive semi-definiteness and unit trace.
        Alert the user about violations of the specific properties.
        """
        herm_flag = np.allclose(self.matrix, self.matrix.T.conj())
        pos_flag = np.allclose(np.minimum(np.real(la.eigvals(self.matrix)), 0), 0)
        trace_flag = np.allclose(np.trace(self.matrix), 1)
        if herm_flag and pos_flag and trace_flag:
            return True
        if not herm_flag:
            print('Non-hermitian', file=sys.stderr)
        if not pos_flag:
            print('Non-positive', file=sys.stderr)
        if not trace_flag:
            print('Trace is not 1', file=sys.stderr)
        return False

    def is_pure(self):
        """Check if the quantum object is a valid rank-1 density matrix"""
        return (np.linalg.matrix_rank(self.matrix, tol=1e-10, hermitian=True) == 1) and self.is_density_matrix()

    def copy(self):
        """Create a copy of this Qobj instance"""
        return deepcopy(self)

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
