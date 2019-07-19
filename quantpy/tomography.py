import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

from .geometry import hs_dst, if_dst, trace_dst, product
from .qobj import Qobj
from .measurements import generate_measurement_matrix


# TODO: fix bloch vector
def _is_feasible(bloch_vector):
    """Constraint for minimize function"""
    return 1 - la.norm(bloch_vector, ord=2)


class Tomograph:
    """Basic class for QST"""
    def __init__(self, state, dst='hs'):
        """
        Initialize quantum state tomography procedure with the preferred set_of_POVMs
        Input:
            set_of_POVMs -- like in state_to_proba function (list of POVMs)
        """
        self.POVM_matrix = np.empty((0, 4 ** state.dim))
        self.results = []
        self.n_measurements = 0
        self.state = state
        if isinstance(dst, str):
            if dst == 'hs':
                self.dst = hs_dst
            elif dst == 'trace':
                self.dst = trace_dst
            elif dst == 'if':
                self.dst = if_dst
            else:
                raise ValueError('Invalid value for argument `dst`')
        else:
            self.dst = dst

    def experiment(self, n_measurements, POVM='mub', warm_start=False):
        POVM_matrices = generate_measurement_matrix(POVM, self.state.dim)
        if isinstance(n_measurements, int):
            n_measurements = [n_measurements] * len(POVM_matrices)
        for POVM_matrix, n_measurements_elem in zip(POVM_matrices, n_measurements):
            self._experiment(POVM_matrix, n_measurements_elem)

    def _experiment(self, POVM_matrix, n_measurements):
        probas = POVM_matrix @ self.state.bloch
        self.results.append(np.random.multinomial(n_measurements, probas))
        self.n_measurements += n_measurements

    def point_estimate(self, method='lin', physical=True):
        if method == 'lin':
            return self._point_estimate_lin(physical=physical)
        else:
            return self._point_estimate_mle(physical=physical)

    def bootstrap_state(self, state, n_measurements, n_repeats, method='lin', dst='hs'):
        pass

    def _point_estimate_lin(self, physical):
        pass

    def _point_estimate_mle(self, physical=True):
        pass
