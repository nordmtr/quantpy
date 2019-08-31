import numpy as np


def generate_measurement_matrix(POVM='proj', n_qubits=1):
    """Generates POVM matrix.

    Parameters
    ----------
    n_qubits : int, default=1
        Number of qubits
    POVM : str or numpy 2-D array, default='proj'
        A single string or a numpy array to construct a POVM matrix.

        Possible strings:
            'proj' -- orthogonal projective measurement, 6^n_qubits rows
            'proj4' -- same as `proj`, but without |-> and |-i> states
            'sic' -- SIC POVM for 1-qubit systems and its tensor products for higher dimensions, 4^n_qubits rows

        Possible numpy arrays:
            2-D array with shape (*, 4) -- interpreted as POVM matrix for 1 qubit,
            construct a POVM matrix for the whole system from tensor products of rows of this matrix
            2-D array with shape (*, 4^n_qubits) -- returns this matrix without any changes

    Returns
    -------
    POVM_matrix : numpy 2-D array
        Matrix of POVM, where each row represents an operator in the Bloch representation
        and rows sum into unity.
    """
    is_full_POVM = False
    if isinstance(POVM, str):
        if POVM == 'proj':
            x_pos = np.array([1, 1, 0, 0])
            y_pos = np.array([1, 0, 1, 0])
            z_pos = np.array([1, 0, 0, 1])
            x_neg = np.array([1, -1, 0, 0])
            y_neg = np.array([1, 0, -1, 0])
            z_neg = np.array([1, 0, 0, -1])
            POVM_1 = np.array([x_pos, x_neg, y_pos, y_neg, z_pos, z_neg]) / 6
        elif POVM == 'proj4':
            x_pos = np.array([1, 1, 0, 0])
            y_pos = np.array([1, 0, 1, 0])
            z_pos = np.array([1, 0, 0, 1])
            z_neg = np.array([1, 0, 0, -1])
            POVM_1 = np.array([x_pos, y_pos, z_pos, z_neg]) / 4
        elif POVM == 'sic':
            sq3 = 1 / np.sqrt(3)
            a0 = np.array([1, sq3, sq3, sq3])
            a1 = np.array([1, sq3, -sq3, -sq3])
            a2 = np.array([1, -sq3, sq3, -sq3])
            a3 = np.array([1, -sq3, -sq3, sq3])
            POVM_1 = np.array([a0, a1, a2, a3]) / 4
        else:
            raise ValueError('Incorrect string shortcut for argument `POVM`')
    elif isinstance(POVM, np.ndarray):
        if POVM.shape[1] == 4:
            POVM_1 = POVM
        elif POVM.shape[1] == 4 ** n_qubits:
            POVM_matrix = POVM
            is_full_POVM = True
        else:
            raise ValueError('Incorrect POVM matrix')
    else:
        raise ValueError('Incorrect value for argument `POVM`')
    if not is_full_POVM:
        POVM_matrix = POVM_1
        for _ in range(n_qubits - 1):
            POVM_matrix = np.kron(POVM_matrix, POVM_1)
    return POVM_matrix
