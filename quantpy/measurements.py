import numpy as np


def generate_measurement_matrix(povm="proj", n_qubits=1):
    """Generates POVM matrix.

    Parameters
    ----------
    n_qubits : int, default=1
        Number of qubits
    povm : str or numpy 2-D or 3-D array, default='proj'
        A single string or a numpy array to construct a POVM matrix.

        Possible strings:
            'proj' -- random orthogonal projective measurement, 6^n_qubits rows
            'proj-set' -- true orthogonal projective measurement, set of POVMs
            'proj4' -- same as `proj`, but without |-> and |-i> states
            'sic' -- SIC POVM for 1-qubit systems and its tensor products for higher dimensions,
                     4^n_qubits rows

        Possible numpy arrays:
            2-D array with shape (*, 4) -- interpreted as POVM matrix for 1 qubit,
            construct a POVM matrix for the whole system from tensor products of rows of this matrix
            3-D array with shape (*, *, 4) -- same, but set of POVMs
            2-D array with shape (*, 4^n_qubits) -- returns this matrix without any changes
            3-D array with shape (*, *, 4^n_qubits) -- same, but set of POVMs

    Returns
    -------
    povm_matrix : numpy 3-D array
        Set of matrices of POVMs, where each row represents an operator in the Bloch representation
        and rows sum into unity.
    """
    is_full_povm = False
    if isinstance(povm, str):
        if povm == "proj":
            x_pos = np.array([1, 1, 0, 0])
            x_neg = np.array([1, -1, 0, 0])
            y_pos = np.array([1, 0, 1, 0])
            y_neg = np.array([1, 0, -1, 0])
            z_pos = np.array([1, 0, 0, 1])
            z_neg = np.array([1, 0, 0, -1])
            povm_1 = np.array([x_pos, x_neg, y_pos, y_neg, z_pos, z_neg]) / 6
        elif povm == "proj-set":
            x_pos = np.array([1, 1, 0, 0])
            x_neg = np.array([1, -1, 0, 0])
            y_pos = np.array([1, 0, 1, 0])
            y_neg = np.array([1, 0, -1, 0])
            z_pos = np.array([1, 0, 0, 1])
            z_neg = np.array([1, 0, 0, -1])
            povm_1 = (
                np.array(
                    [
                        np.array([x_pos, x_neg]),
                        np.array([y_pos, y_neg]),
                        np.array([z_pos, z_neg]),
                    ]
                )
                / 2
            )
        elif povm == "proj4":
            x_pos = np.array([1, 1, 0, 0])
            y_pos = np.array([1, 0, 1, 0])
            z_pos = np.array([1, 0, 0, 1])
            z_neg = np.array([1, 0, 0, -1])
            povm_1 = np.array([x_pos, y_pos, z_pos, z_neg]) / 4
        elif povm == "sic":
            sq3 = 1 / np.sqrt(3)
            a0 = np.array([1, sq3, sq3, sq3])
            a1 = np.array([1, sq3, -sq3, -sq3])
            a2 = np.array([1, -sq3, sq3, -sq3])
            a3 = np.array([1, -sq3, -sq3, sq3])
            povm_1 = np.array([a0, a1, a2, a3]) / 4
        else:
            raise ValueError("Incorrect string shortcut for argument `povm`")
    elif isinstance(povm, np.ndarray):
        if povm.shape[-1] == 4:
            povm_1 = povm
        elif povm.shape[-1] == 4**n_qubits:
            if len(povm.shape) == 2:
                povm = povm[None, :, :]
            povm_matrix = povm
            is_full_povm = True
        else:
            raise ValueError("Incorrect POVM matrix")
    else:
        raise ValueError("Incorrect value for argument `povm`")
    if not is_full_povm:
        if len(povm_1.shape) == 2:
            povm_1 = povm_1[None, :, :]
        povm_matrix = povm_1
        for _ in range(n_qubits - 1):
            povm_matrix = np.kron(povm_matrix, povm_1)
    return povm_matrix
