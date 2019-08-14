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
    EPS = 1e-15
    v, U = la.eigh(qobj.matrix)
    V = np.diag(np.maximum(EPS, v))  # positiveness
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

    def point_estimate(self, method='lin', physical=True, init='mixed'):
        if method == 'lin':
            self.reconstructed_state = self._point_estimate_lin(physical=physical)
        elif method == 'mle-chol':
            self.reconstructed_state = self._point_estimate_mle_chol(init=init)
        elif method == 'mle-chol-constr':
            self.reconstructed_state = self._point_estimate_mle_chol_constr(init=init)
        elif method == 'mle-bloch':
            self.reconstructed_state = self._point_estimate_mle_bloch(physical=physical)
        else:
            raise ValueError('Invalid value for argument `method`')
        return self.reconstructed_state

    def bootstrap(self, n_measurements, n_repeats,
                  est_method='lin', physical=True, init='mixed', use_new_estimate=False):
        """
        Perform quantum tomography with *n_measurements* on reconstructed state from results *n_repeats* times
        Output:
            Sorted list of distances between the input state and corresponding estimated matrices
        """
        if use_new_estimate:
            state = self.point_estimate(method=est_method, physical=physical, init=init)
        else:
            state = self.reconstructed_state
        dist = [0]
        boot_tmg = self.__class__(state, self.dst)
        for _ in range(n_repeats):
            boot_tmg.experiment(n_measurements, POVM=self.POVM_matrix)
            rho = boot_tmg.point_estimate(method=est_method, physical=physical, init=init)
            dist.append(self.dst(rho, state))
        dist.sort()
        return dist

    def _point_estimate_lin(self, physical):
        frequencies = self.results / self.results.sum()
        bloch_vec = _left_inv(self.POVM_matrix) @ frequencies / (2 ** self.state.dim)
        rho = Qobj(bloch_vec)
        if physical:
            rho = _make_feasible(rho)
        return rho

    def _point_estimate_mle_chol(self, init):
        if init == 'mixed':
            x0 = np.eye(2 ** self.state.dim) / (2 ** self.state.dim)  # fully mixed state
        elif init == 'lin':
            x0 = self.point_estimate('lin').matrix
        else:
            raise ValueError('Invalid value for argument `init`')
        x0 = _matrix_to_real_tril_vec(x0)
        opt_res = minimize(self._neg_log_likelihood_chol, x0, method='BFGS')
        matrix = _real_tril_vec_to_matrix(opt_res.x)
        return Qobj(matrix / np.trace(matrix))

    def _neg_log_likelihood_chol(self, tril_vec):
        EPS = 1e-10
        matrix = _real_tril_vec_to_matrix(tril_vec)
        rho = Qobj(matrix / np.trace(matrix))
        probas = self.POVM_matrix @ rho.bloch * (2 ** self.state.dim)
        log_likelihood = np.sum(self.results * np.log(probas + EPS))
        return -log_likelihood

    def _point_estimate_mle_chol_constr(self, init):
        constraints = [
            {'type': 'eq', 'fun': _is_unit_trace},
        ]
        if init == 'mixed':
            x0 = np.eye(2 ** self.state.dim) / (2 ** self.state.dim)  # fully mixed state
        elif init == 'lin':
            x0 = self.point_estimate('lin').matrix
        else:
            raise ValueError('Invalid value for argument `init`')
        x0 = _matrix_to_real_tril_vec(x0)
        opt_res = minimize(self._neg_log_likelihood_chol, x0, constraints=constraints, method='SLSQP')
        matrix = _real_tril_vec_to_matrix(opt_res.x)
        return Qobj(matrix / np.trace(matrix))

    def _neg_log_likelihood_chol_constr(self, tril_vec):
        EPS = 1e-10
        rho = Qobj(_real_tril_vec_to_matrix(tril_vec))
        probas = self.POVM_matrix @ rho.bloch * (2 ** self.state.dim)
        log_likelihood = np.sum(self.results * np.log(probas + EPS))
        return -log_likelihood

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

    def _neg_log_likelihood_bloch(self, bloch_vec):
        EPS = 1e-10
        bloch_vec = np.append(1 / 2 ** self.state.dim, bloch_vec)
        probas = self.POVM_matrix @ bloch_vec * (2 ** self.state.dim)
        log_likelihood = np.sum(self.results * np.log(probas + EPS))
        return -log_likelihood
