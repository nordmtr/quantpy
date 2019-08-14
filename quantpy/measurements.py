import numpy as np

# from .qobj import Qobj
# from .routines import generate_pauli


def generate_measurement_matrix(POVM='proj', dim=1):
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
        elif POVM.shape[1] == 4 ** dim:
            POVM_matrix = POVM
            is_full_POVM = True
        else:
            raise ValueError('Incorrect POVM matrix')
    else:
        raise ValueError('Incorrect value for argument `POVM`')
    if not is_full_POVM:
        POVM_matrix = POVM_1
        for _ in range(dim - 1):
            POVM_matrix = np.kron(POVM_matrix, POVM_1)
    return POVM_matrix
