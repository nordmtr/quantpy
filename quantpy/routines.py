import numpy as np
import scipy.linalg as la

# Pauli basis

_SIGMA_I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
_SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

_PAULI_1 = [_SIGMA_I, _SIGMA_X, _SIGMA_Y, _SIGMA_Z]


def generate_pauli(n_qubits):
    """Generate basis of Pauli matrices for `n_qubits` qubits"""
    basis = _PAULI_1
    for _ in range(n_qubits - 1):
        basis = np.kron(basis, _PAULI_1)
    return basis


def generate_single_entries(dim):
    """Generate all matrices with shape dim * dim with single entries.
    If `to_qobj` is True, return Qobj instead of numpy arrays"""
    list_of_single_entries = []
    for i in range(dim):
        for j in range(dim):
            single_entry = np.array(np.zeros((dim, dim)))
            single_entry[i, j] = 1
            list_of_single_entries.append(single_entry)
    return list_of_single_entries


def kron(A, B):
    """Same as `kron` method for `Qobj` and `Operator` classes"""
    return A.kron(B)


def join_gates(gates):
    """Construct one gate from a list of gates"""
    new_gate = gates[0]
    for gate in gates[1:]:
        new_gate = gate @ new_gate
    return new_gate


def _out_ptrace_oper(n_qubits):
    """Construct a partial trace operator over output space for a bipartite system"""
    identity = np.eye(2**n_qubits)
    return np.sum([np.kron(identity, np.kron(k_vec, np.kron(identity, k_vec))) for k_vec in identity], axis=0)


def _vec2mat(vector):
    """Reconstruct a matrix from the vector using column-stacking convention"""
    shape = int(np.sqrt(len(vector)))
    return vector.reshape(shape, shape).T


def _mat2vec(matrix):
    """Convert the matrix into a vector using column-stacking convention"""
    return matrix.T.reshape(np.prod(matrix.shape))


def _density(psi):
    """Construct a density matrix of a pure state"""
    return np.outer(np.asarray(psi, dtype=np.complex128).T, np.conj(np.asarray(psi, dtype=np.complex128)))


def _left_inv(A):
    """Return left pseudo-inverse matrix"""
    return la.inv(A.T @ A) @ A.T


def _real_to_complex(z):
    """Real vector of length 2n -> complex of length n"""
    return z[: len(z) // 2] + 1j * z[len(z) // 2 :]


def _complex_to_real(z):
    """Complex vector of length n -> real of length 2n"""
    return np.concatenate((np.real(z), np.imag(z)))


def _matrix_to_real_tril_vec(matrix):
    """Parametrize a positive definite hermitian matrix using its Cholesky decomposition"""
    tril_matrix = la.cholesky(matrix, lower=True)
    diag_vector = tril_matrix[np.diag_indices(tril_matrix.shape[0])].astype(float)
    complex_tril_vector = tril_matrix[np.tril_indices(tril_matrix.shape[0], -1)]
    real_tril_vector = _complex_to_real(complex_tril_vector)
    return np.concatenate((diag_vector, real_tril_vector))


def _real_tril_vec_to_matrix(vector):
    """Restore a matrix from its Cholesky parametrization"""
    matrix_shape = int(np.sqrt(len(vector)))  # solve a quadratic equation for the shape
    diag_vector = vector[:matrix_shape]
    complex_vector = _real_to_complex(vector[matrix_shape:])
    tril_matrix = np.zeros((matrix_shape, matrix_shape), dtype=np.complex128)
    tril_matrix[np.tril_indices(matrix_shape, -1)] = complex_vector
    tril_matrix[np.diag_indices(matrix_shape)] = diag_vector
    return tril_matrix @ tril_matrix.T.conj()
