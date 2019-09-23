import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

from ..geometry import hs_dst, if_dst, trace_dst
from ..qobj import Qobj, fully_mixed
from ..measurements import generate_measurement_matrix
from ..routines import _left_inv, _matrix_to_real_tril_vec, _real_tril_vec_to_matrix


def _is_positive(bloch_vec):  # works only for 1-qubit systems !!
    """Positivity constraint for minimize function based on the bloch vector norm"""
    return 0.5 - la.norm(bloch_vec, ord=2)


def _is_unit_trace(tril_vec):
    """Unit trace constraint for minimize function"""
    matrix = _real_tril_vec_to_matrix(tril_vec)
    return np.trace(matrix) - 1


def _make_feasible(qobj):
    """Make the matrix positive semi-definite and with unit trace"""
    EPS = 1e-15
    v, U = la.eigh(qobj.matrix)
    V = np.diag(np.maximum(EPS, v))  # positiveness
    matrix = U @ V @ U.T.conj()
    return Qobj(matrix / np.trace(matrix))


def _make_feasible_bloch(qobj):  # works only for 1-qubit systems !!
    """Normalize the Bloch vector if it is outside the Bloch sphere"""
    bloch_vec = qobj.bloch.copy()
    bloch_norm = la.norm(bloch_vec[1:], ord=2)
    if bloch_norm > 0.5:
        bloch_vec[1:] *= 0.5 / bloch_norm
    bloch_vec[0] = 0.5
    return Qobj(bloch_vec)


class StateTomograph:
    """Basic class for quantum state tomography

    This class supports experiment simulations, different methods of reconstructing a density matrix
    from the data and building confidence intervals.

    Parameters
    ----------
    state : Qobj
        Quantum object to perform a tomography on
    dst : str or callable, default='hs'
        Measure in a space of quantum objects

        Possible strings:
            'hs' -- Hilbert-Schmidt distance
            'trace' -- trace distance
            'if' -- infidelity

        Interface for a custom measure:
            custom_measure(A: Qobj, B: Qobj) -> float

    Attributes
    ----------
    n_measurements : float
        Total number of measurements made during tomography
    POVM_matrix : numpy 2-D array
        Numpy array with shape (*, 4^n_qubits), representing the measurement matrix.
        Rows are bloch vectors that sum into unity
    reconstructed_state : Qobj
        The most recent estimation of a density matrix, if ever performed
    results : numpy 1-D array
        Results of the simulated experiment.
        results[i] is the number of outcomes corresponding to POVM_matrix[i,:]

    Methods
    -------
    bootstrap()
        Perform multiple tomography simulation
    experiment()
        Simulate a real quantum state tomography
    point_estimate()
        Reconstruct a density matrix from the data obtained in the experiment
    """
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
        """Simulate a real quantum state tomography.

        Parameters
        ----------
        n_measurements : int
            Number of measurements to perform in the tomography
        POVM : str or numpy 2-D array, default='proj'
            A single string or a numpy array to construct a POVM matrix.

            Possible strings:
                'proj' -- orthogonal projective measurement, 6^n_qubits rows
                'sic' -- SIC POVM for 1-qubit systems and its tensor products for higher dimensions, 4^n_qubits rows

            Possible numpy arrays:
                2-D array with shape (*, 4) -- interpreted as POVM matrix for 1 qubit,
                construct a POVM matrix for the whole system from tensor products of rows of this matrix
                2-D array with shape (*, 4^n_qubits) -- returns this matrix without any changes

            See :ref:`generate_measurement_matrix` for more detailed documentation

        warm_start : bool, default=False
            If True, do not overwrite the previous experiment results, add all results to those of the previous run
        """
        POVM_matrix = generate_measurement_matrix(POVM, self.state.n_qubits)
        probas = POVM_matrix @ self.state.bloch * (2 ** self.state.n_qubits)
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

    def point_estimate(self, method='lin', physical=True, init='lin'):
        """Reconstruct a density matrix from the data obtained in the experiment

        Parameters
        ----------
        method : str, default='lin'
            Method of reconstructing the density matrix

            Possible values:
                'lin' -- linear inversion
                'mle' -- maximum likelihood estimation with Cholesky parametrization, unconstrained optimization
                'mle-constr' -- same as 'mle', but optimization is constrained
                'mle-bloch' -- maximum likelihood estimation with Bloch parametrization,
                               constrained optimization (works only for 1-qubit systems)

        physical : bool, default=True (optional)
            For methods 'lin' and 'mle' reconstructed matrix may not lie in the physical domain.
            If True, set negative eigenvalues to zeros and divide the matrix by its trace.

        init : str, default='lin' (optional)
            Methods using maximum likelihood estimation require the starting point for gradient descent.

            Possible values:
                'lin' -- uses linear inversion point estimate as initial guess
                'mixed' -- uses fully mixed state as initial guess

        Returns
        -------
        reconstructed_state : Qobj
        """
        if method == 'lin':
            self.reconstructed_state = self._point_estimate_lin(physical=physical)
        elif method == 'mle':
            self.reconstructed_state = self._point_estimate_mle_chol(init=init)
        elif method == 'mle-constr':
            self.reconstructed_state = self._point_estimate_mle_chol_constr(init=init)
        elif method == 'mle-bloch':
            self.reconstructed_state = self._point_estimate_mle_bloch(physical=physical)
        else:
            raise ValueError('Invalid value for argument `method`')
        return self.reconstructed_state

    def bootstrap(self, n_boot, est_method='lin', physical=True, init='lin',
                  use_new_estimate=False, state=None, kind='estim'):
        """Perform multiple tomography simulation on the preferred state with the same measurements number
        and POVM matrix, as in the preceding experiment. Count the distances to the bootstrapped states.

        Parameters
        ----------
        n_boot : int
            Number of experiments to perform
        est_method : str, default='lin'
            Method of reconstructing the density matrix
            See :ref:`point_estimate` for detailed documentation
        physical : bool, default=True (optional)
            See :ref:`point_estimate` for detailed documentation
        init : str, default='lin' (optional)
            See :ref:`point_estimate` for detailed documentation
        use_new_estimate : bool, default=False
            If False, uses the latest reconstructed state as a state to perform new tomographies on.
            If True and `state` is None, reconstruct a density matrix from the data obtained in previous experiment
            ans use it to perform new tomographies on.
            If True and `state` is not None, use `state` as a state to perform new tomographies on.
        state : Qobj or None, default=None
            If not None and `use_new_estimate` is True, use it as a state to perform new tomographies on
        kind : str, default='est'
            Type of confidence interval to build.
            Possible values:
                'estim' -- CI for the point estimate
                'target' -- CI for the target state built with bootstrap from point estimate only
                'triangle' -- CI for the target state built with bootstrap from point estimate only
                              + triangle inequality
        """
        if not use_new_estimate:
            state = self.reconstructed_state
        elif state is None:
            state = self.point_estimate(method=est_method, physical=physical, init=init)

        dist = np.zeros(n_boot + 1)
        boot_tmg = self.__class__(state, self.dst)
        for i in range(n_boot):
            boot_tmg.experiment(self.n_measurements, POVM=self.POVM_matrix)
            rho = boot_tmg.point_estimate(method=est_method, physical=physical, init=init)
            if kind == 'estim':
                dist[i + 1] = self.dst(rho, state)
            elif kind == 'target':
                dist[i + 1] = self.dst(rho, self.state)
            elif kind == 'triangle':
                dist[i + 1] = self.dst(rho, state) + self.dst(state, self.state)
            else:
                raise ValueError('Invalid value for argument `kind`')
        dist.sort()
        return dist

    def _point_estimate_lin(self, physical):
        """Point estimate based on linear inversion algorithm"""
        frequencies = self.results / self.results.sum()
        bloch_vec = _left_inv(self.POVM_matrix) @ frequencies / (2 ** self.state.n_qubits)
        rho = Qobj(bloch_vec)
        if physical:
            rho = _make_feasible(rho)
        return rho

    def _point_estimate_mle_chol(self, init):
        """Point estimate based on MLE with Cholesky parametrization"""
        if init == 'mixed':
            x0 = fully_mixed(self.state.n_qubits).matrix
        elif init == 'lin':
            x0 = self.point_estimate('lin').matrix
        else:
            raise ValueError('Invalid value for argument `init`')
        x0 = _matrix_to_real_tril_vec(x0)
        opt_res = minimize(self._neg_log_likelihood_chol, x0, method='BFGS')
        matrix = _real_tril_vec_to_matrix(opt_res.x)
        return Qobj(matrix / np.trace(matrix))

    def _neg_log_likelihood_chol(self, tril_vec):
        """Negative log-likelihood for MLE with Cholesky parametrization"""
        EPS = 1e-10
        matrix = _real_tril_vec_to_matrix(tril_vec)
        rho = Qobj(matrix / np.trace(matrix))
        probas = self.POVM_matrix @ rho.bloch * (2 ** self.state.n_qubits)
        log_likelihood = np.sum(self.results * np.log(probas + EPS)) / self.n_measurements
        return -log_likelihood

    def _point_estimate_mle_chol_constr(self, init):
        """Point estimate based on constrained MLE with Cholesky parametrization"""
        constraints = [
            {'type': 'eq', 'fun': _is_unit_trace},
        ]
        if init == 'mixed':
            x0 = fully_mixed(self.state.n_qubits).matrix
        elif init == 'lin':
            x0 = self.point_estimate('lin').matrix
        else:
            raise ValueError('Invalid value for argument `init`')
        x0 = _matrix_to_real_tril_vec(x0)
        opt_res = minimize(self._neg_log_likelihood_chol, x0, constraints=constraints, method='SLSQP')
        matrix = _real_tril_vec_to_matrix(opt_res.x)
        return Qobj(matrix / np.trace(matrix))

    def _neg_log_likelihood_chol_constr(self, tril_vec):
        """Negative log-likelihood for constrained MLE with Cholesky parametrization"""
        EPS = 1e-10
        rho = Qobj(_real_tril_vec_to_matrix(tril_vec))
        probas = self.POVM_matrix @ rho.bloch * (2 ** self.state.n_qubits)
        log_likelihood = np.sum(self.results * np.log(probas + EPS)) / self.n_measurements
        return -log_likelihood

    def _point_estimate_mle_bloch(self, physical):  # works only for 1-qubit systems
        """Point estimate based on MLE with Bloch parametrization"""
        constraints = [
            {'type': 'ineq', 'fun': _is_positive},
        ]
        x0 = np.zeros(4 ** self.state.n_qubits - 1)  # fully mixed state
        opt_res = minimize(self._neg_log_likelihood_bloch, x0, constraints=constraints, method='SLSQP')
        bloch_vec = np.append(1 / 2 ** self.state.n_qubits, opt_res.x)
        rho = Qobj(bloch_vec)
        if physical:
            rho = self._make_feasible(rho)
        return rho

    def _neg_log_likelihood_bloch(self, bloch_vec):
        """Negative log-likelihood for MLE with Bloch parametrization"""
        EPS = 1e-10
        bloch_vec = np.append(1 / 2 ** self.state.n_qubits, bloch_vec)
        probas = self.POVM_matrix @ bloch_vec * (2 ** self.state.n_qubits)
        log_likelihood = np.sum(self.results * np.log(probas + EPS)) / self.n_measurements
        return -log_likelihood