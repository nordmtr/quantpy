import math
import sys
from copy import deepcopy

import numpy as np
import scipy.linalg as la

from .base_quantum import BaseQuantum
from .geometry import product
from .routines import _density, generate_pauli


class Qobj(BaseQuantum):
    """Basic class for representing quantum objects, such as quantum states and measurement
    operators

    This class supports all simple math operations, as well as a collection of common
    state/operator operations.

    Parameters
    ----------
    data : array-like or None, default=None
        If `data` is 2-D, it is treated as a full matrix
        If `data` is 1-D and `is_ket` is False, it is treated as a bloch vector
        If `data` is 1-D and `is_let` is True, it is treated as a ket vector
    is_ket : bool, default=False

    Attributes
    ----------
    bloch : numpy 1-D array (property)
        A vector, representing the quantum object in Pauli basis (only for Hermitian matrices)
    H : Qobj (property)
        Adjoint matrix of the quantum object
    matrix : numpy 2-D array (property)
        Quantum object in a matrix form
    n_qubits : int
        Number of qubits
    T : Qobj (property)
        Transpose of the quantum object

    Methods
    -------
    conj()
        Conjugate of the quantum object
    copy()
        Create a copy of this Qobj instance
    eig()
        Eigenvalues and eigenvectors of the quantum object
    is_density_matrix()
        Check if the quantum object is valid density matrix
    is_pure()
        Check if the quantum object is rank-1 valid density matrix
    impurity()
        Return impurity measure 1-Tr(rho^2)
    ket() : list
        Ket vector representation of the quantum object
    kron()
        Kronecker product of 2 Qobj instances
    ptrace()
        Partial trace of the quantum object
    schmidt()
        Schmidt decomposition of the quantum object
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

    def __init__(self, data, is_ket=False):
        if isinstance(data, self.__class__):
            self.__dict__ = deepcopy(data.__dict__)
        else:
            self._types = set()  # Set of types which represent the state
            if is_ket:
                data = _density(data)
            data = np.array(data)
            if len(data.shape) == 1:
                n_qubits_float = math.log2(data.shape[0]) / 2
                self.n_qubits = math.ceil(n_qubits_float)
                dim = 2**self.n_qubits
                if n_qubits_float.is_integer():
                    self._bloch = data
                else:
                    self._bloch = np.ones(dim**2) / dim
                    self._bloch[1:] = data
                self._matrix = None
                self._types.add("bloch")
            elif len(data.shape) == 2:
                self._matrix = data
                self._bloch = None
                self._types.add("matrix")
                self.n_qubits = int(np.log2(data.shape[0]))
            else:
                raise ValueError("Invalid data format")

    @property
    def matrix(self):
        """Quantum object in a matrix form"""
        if "matrix" not in self._types:
            self._types.add("matrix")
            basis = generate_pauli(self.n_qubits)
            self._matrix = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=np.complex128)
            for i in range(4**self.n_qubits):
                self._matrix += basis[i] * self._bloch[i]
        return self._matrix

    @matrix.setter
    def matrix(self, data):
        self._types.add("matrix")
        self._types.discard("bloch")
        self._matrix = np.array(data)

    @property
    def bloch(self):
        """A vector, representing the quantum object in Pauli basis"""
        if "bloch" not in self._types:
            self._types.add("bloch")
            basis = generate_pauli(self.n_qubits)
            self._bloch = np.array([np.real(product(basis_element, self._matrix)) for basis_element in basis]) / (
                2**self.n_qubits
            )
        return self._bloch

    @bloch.setter
    def bloch(self, data):
        if isinstance(data, list):
            data = np.array(data)
        self._types.add("bloch")
        self._types.discard("matrix")
        self._bloch = np.array(data)

    def ptrace(self, keep=(0,)):
        """Partial trace of the quantum object

        Parameters
        ----------
        keep : array-like, default=[0]
            List of indices of subsystems to keep after being traced.

        Returns
        -------
        rho : Qobj
            Traced quantum object
        """
        keep = np.array(keep)

        bra_idx = list(range(self.n_qubits))
        # preserve indices in `keep`
        ket_idx = [self.n_qubits + i if i in keep else i for i in range(self.n_qubits)]
        rho = self.matrix.reshape([2] * (2 * self.n_qubits))
        rho = np.einsum(rho, bra_idx + ket_idx)  # sum over the preferred indices
        return Qobj(rho.reshape(2 ** len(keep), 2 ** len(keep)))

    def schmidt(self):
        """Return Schmidt decomposition of the quantum object, if it is pure and consists of 2
        subsystems.

        Returns
        -------
        U : complex numpy 2-D array
            Unitary matrix having first subsystem vectors as columns
        s : complex numpy 1-D array
            Singular values of the decomposition, sorted in non-increasing order
        Vh : complex 2-D array
            Unitary matrix having second subsystem vectors as rows
        """
        matrix_dim = 2 ** int(self.n_qubits / 2)
        matrix_repr = np.reshape(self.ket(), (matrix_dim, matrix_dim))
        return la.svd(matrix_repr)

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

    def is_density_matrix(self, verbose=True):
        """Check if the quantum object is a valid density matrix.
        Perform a test for hermiticity, positive semi-definiteness and unit trace.
        Alert the user about violations of the specific properties.
        """
        herm_flag = np.allclose(self.matrix, self.matrix.T.conj())
        pos_flag = np.allclose(np.minimum(np.real(self.eig()[0]), 0), 0)
        trace_flag = np.allclose(np.trace(self.matrix), 1)
        if herm_flag and pos_flag and trace_flag:
            return True
        if not herm_flag and verbose:
            print("Non-hermitian", file=sys.stderr)
        if not pos_flag and verbose:
            print("Non-positive", file=sys.stderr)
        if not trace_flag and verbose:
            print("Trace is not 1", file=sys.stderr)
        return False

    def trace(self):
        """Trace of the quantum object"""
        return np.trace(self.matrix)

    def impurity(self):
        """Return impurity measure 1-Tr(rho^2)"""
        return 1 - (self @ self).trace()

    def is_pure(self):
        """Check if the quantum object is a valid rank-1 density matrix"""
        return np.allclose(self.impurity(), 0) and self.is_density_matrix()

    def ket(self):
        """Return ket vector representation of the quantum object if it is pure"""
        if not self.is_pure():
            raise ValueError("Quantum object is not pure")
        return self.eig()[1][:, 0]

    def __repr__(self):
        return "Quantum object\n" + repr(self.matrix)

    def _repr_latex_(self):
        """Generate a LaTeX representation of the Qobj instance. Can be used for
        formatted output in IPython notebook.
        """
        s = r"Quantum object: "
        M, N = self.matrix.shape

        s += r"\begin{equation*}\left(\begin{array}{*{11}c}"

        def _format_float(value):
            if value == 0.0:
                return "0.0"
            elif abs(value) > 1000.0 or abs(value) < 0.001:
                return ("%.3e" % value).replace("e", r"\times10^{") + "}"
            elif abs(value - int(value)) < 0.001:
                return "%.1f" % value
            else:
                return "%.3f" % value

        def _format_element(m, n, d):
            s = " & " if n > 0 else ""
            if type(d) == str:
                return s + d
            else:
                atol = 1e-4
                if abs(np.imag(d)) < atol:
                    return s + _format_float(np.real(d))
                elif abs(np.real(d)) < atol:
                    return s + _format_float(np.imag(d)) + "j"
                else:
                    s_re = _format_float(np.real(d))
                    s_im = _format_float(np.imag(d))
                    if np.imag(d) > 0.0:
                        return s + "(" + s_re + "+" + s_im + "j)"
                    else:
                        return s + "(" + s_re + s_im + "j)"

        if M > 10 and N > 10:
            # truncated matrix output
            for m in range(5):
                for n in range(5):
                    s += _format_element(m, n, self.matrix[m, n])
                s += r" & \cdots"
                for n in range(N - 5, N):
                    s += _format_element(m, n, self.matrix[m, n])
                s += r"\\"

            for n in range(5):
                s += _format_element(m, n, r"\vdots")
            s += r" & \ddots"
            for n in range(N - 5, N):
                s += _format_element(m, n, r"\vdots")
            s += r"\\"

            for m in range(M - 5, M):
                for n in range(5):
                    s += _format_element(m, n, self.matrix[m, n])
                s += r" & \cdots"
                for n in range(N - 5, N):
                    s += _format_element(m, n, self.matrix[m, n])
                s += r"\\"

        elif M > 10 and N <= 10:
            # truncated vertically elongated matrix output
            for m in range(5):
                for n in range(N):
                    s += _format_element(m, n, self.matrix[m, n])
                s += r"\\"

            for n in range(N):
                s += _format_element(m, n, r"\vdots")
            s += r"\\"

            for m in range(M - 5, M):
                for n in range(N):
                    s += _format_element(m, n, self.matrix[m, n])
                s += r"\\"

        elif M <= 10 and N > 10:
            # truncated horizontally elongated matrix output
            for m in range(M):
                for n in range(5):
                    s += _format_element(m, n, self.matrix[m, n])
                s += r" & \cdots"
                for n in range(N - 5, N):
                    s += _format_element(m, n, self.matrix[m, n])
                s += r"\\"

        else:
            # full output
            for m in range(M):
                for n in range(N):
                    s += _format_element(m, n, self.matrix[m, n])
                s += r"\\"

        s += r"\end{array}\right)\end{equation*}"
        return s


def fully_mixed(n_qubits=1):
    """Return fully mixed state."""
    dim = 2**n_qubits
    return Qobj(np.eye(dim, dtype=np.complex128) / dim)


# noinspection PyPep8Naming
def GHZ(n_qubits=3):
    """Return GHZ state."""
    ket = ([1] + [0] * (2**n_qubits - 2) + [1]) / np.sqrt(2)
    return Qobj(ket, is_ket=True)


def zero(n_qubits=1):
    """Return zero state."""
    ket = [1] + [0] * (2**n_qubits - 1)
    return Qobj(ket, is_ket=True)
