import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

from .geometry import hs_dst, if_dst, trace_dst
from .qobj import Qobj
from .measurements import generate_measurement_matrix
from .routines import _left_inv, _matrix_to_real_tril_vec, _real_tril_vec_to_matrix


def _is_positive(bloch_vec):  # works only for 1-qubit systems !!
    """Positivity constraint for minimize function based on the trace condition"""
    bloch_len = len(bloch_vec) + 1  # 4 ** dim
    return np.sqrt(bloch_len) - 1 - bloch_len * np.sum(bloch_vec ** 2)


def _is_unit_trace(tril_vec):
    """Unit trace constraint for minimize function"""
    matrix = _real_tril_vec_to_matrix(tril_vec)
    return np.trace(matrix) - 1


def _make_feasible(qobj):
    v, U = la.eigh(qobj.matrix)
    V = np.diag(np.maximum(0, v))  # positiveness
    matrix = U @ V @ la.inv(U)
    return Qobj(matrix / np.trace(matrix))


class Tomograph:
    """Basic class for QST"""
    def __init__(self, state, dst='hs'):
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

    def experiment(self, n_measurements, POVM='proj', warm_start=False):
        POVM_matrix = generate_measurement_matrix(POVM, self.state.dim)
        probas = POVM_matrix @ self.state.bloch * (2 ** self.state.dim)
        results = np.random.multinomial(n_measurements, probas)
        if warm_start:
            self.POVM_matrix = np.vstack((
                self.POVM_matrix * self.n_measurements,
                POVM_matrix * n_measurements,
            )) / (self.n_measurements + n_measurements)
            self.results = np.hstack((self.results, results))
            self.n_measurements += n_measurements
        else:
            self.POVM_matrix = POVM_matrix
            self.results = results
            self.n_measurements = n_measurements

    def point_estimate(self, method='lin', physical=True):
        if method == 'lin':
            return self._point_estimate_lin(physical=physical)
        elif method == 'mle-chol':
            return self._point_estimate_mle_chol()
        elif method == 'mle-chol-constr':
            return self._point_estimate_mle_chol_constr()
        elif method == 'mle-bloch':
            return self._point_estimate_mle_bloch(physical=physical)
        else:
            raise ValueError('Unknown identifier for argument `method`')

    def bootstrap_state(self, state, n_measurements, n_repeats, method='lin', dst='hs'):
        pass

    def _point_estimate_lin(self, physical):
        frequencies = self.results / self.results.sum()
        bloch_vec = _left_inv(self.POVM_matrix) @ frequencies / (2 ** self.state.dim)
        rho = Qobj(bloch_vec)
        if physical:
            rho = _make_feasible(rho)
        return Qobj(bloch_vec)

    def _point_estimate_mle_bloch(self, physical):  # works only for 1-dimensional systems
        constraints = [
            {'type': 'ineq', 'fun': _is_positive},
        ]
        x0 = np.zeros(4 ** self.state.dim - 1)  # fully mixed state
        opt_res = minimize(self._neg_log_likelihood_bloch, x0, constraints=constraints, method='SLSQP')
        bloch_vec = np.append(1 / 2 ** self.state.dim, opt_res.x)
        rho = Qobj(bloch_vec)
        if physical:
            rho = self._make_feasible(rho)
        return rho

    def _point_estimate_mle_chol(self):
        fully_mixed_state = np.eye(2 ** self.state.dim) / (2 ** self.state.dim)
        x0 = _matrix_to_real_tril_vec(fully_mixed_state)
        opt_res = minimize(self._neg_log_likelihood_chol, x0, method='BFGS')
        matrix = _real_tril_vec_to_matrix(opt_res.x)
        return Qobj(matrix / np.trace(matrix))

    def _point_estimate_mle_chol_constr(self):
        constraints = [
            {'type': 'eq', 'fun': _is_unit_trace},
        ]
        fully_mixed_state = np.eye(2 ** self.state.dim) / (2 ** self.state.dim)
        x0 = _matrix_to_real_tril_vec(fully_mixed_state)
        opt_res = minimize(self._neg_log_likelihood_chol, x0, constraints=constraints, method='SLSQP')
        matrix = _real_tril_vec_to_matrix(opt_res.x)
        return Qobj(matrix / np.trace(matrix))

    def _neg_log_likelihood_bloch(self, bloch_vec):
        EPS = 1e-10
        bloch_vec = np.append(1 / 2 ** self.state.dim, bloch_vec)
        probas = self.POVM_matrix @ bloch_vec * (2 ** self.state.dim)
        log_likelihood = np.sum(self.results * np.log(probas + EPS))
        return -log_likelihood

    def _neg_log_likelihood_chol_constr(self, tril_vec):
        EPS = 1e-10
        rho = Qobj(_real_tril_vec_to_matrix(tril_vec))
        probas = self.POVM_matrix @ rho.bloch * (2 ** self.state.dim)
        log_likelihood = np.sum(self.results * np.log(probas + EPS))
        return -log_likelihood

    def _neg_log_likelihood_chol(self, tril_vec):
        EPS = 1e-10
        matrix = _real_tril_vec_to_matrix(tril_vec)
        rho = Qobj(matrix / np.trace(matrix))
        probas = self.POVM_matrix @ rho.bloch * (2 ** self.state.dim)
        log_likelihood = np.sum(self.results * np.log(probas + EPS))
        return -log_likelihood
