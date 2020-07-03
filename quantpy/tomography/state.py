import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

from ..geometry import hs_dst, if_dst, trace_dst
from ..qobj import Qobj, fully_mixed
from ..measurements import generate_measurement_matrix
from ..routines import _left_inv, _matrix_to_real_tril_vec, _real_tril_vec_to_matrix
from ..mhmc import MHMC, normalized_update


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

    def experiment(self, n_measurements, POVM='proj-set', warm_start=False):
        """Simulate a real quantum state tomography.

        Parameters
        ----------
        n_measurements : int or array-like
            Number of measurements to perform in the tomography
        POVM : str or numpy 2-D array, default='proj'
            A single string or a numpy array to construct a POVM matrix.

            Possible strings:
                'proj' -- random orthogonal projective measurement, 6^n_qubits rows
                'proj-set' -- true orthogonal projective measurement, set of POVMs
                'sic' -- SIC POVM for 1-qubit systems and its tensor products for higher dimensions, 4^n_qubits rows

            Possible numpy arrays:
                2-D array with shape (*, 4) -- interpreted as POVM matrix for 1 qubit,
                construct a POVM matrix for the whole system from tensor products of rows of this matrix
                3-D array with shape (*, *, 4) -- same, but set of POVMs
                2-D array with shape (*, 4^n_qubits) -- returns this matrix without any changes
                3-D array with shape (*, *, 4^n_qubits) -- same, but set of POVMs

            See :ref:`generate_measurement_matrix` for more detailed documentation

        warm_start : bool, default=False
            If True, do not overwrite the previous experiment results, add all results to those of the previous run
        """
        POVM_matrix = generate_measurement_matrix(POVM, self.state.n_qubits)
        number_of_POVMs = POVM_matrix.shape[0]

        if isinstance(n_measurements, int):
            n_measurements = np.ones(number_of_POVMs) * n_measurements
        elif len(n_measurements) != number_of_POVMs:
            raise ValueError('Wrong length for argument `n_measurements`')

        probas = np.einsum('ijk,k->ij', POVM_matrix, self.state.bloch) * (2 ** self.state.n_qubits)
        results = [np.random.multinomial(n_measurements_for_POVM, probas_for_POVM)
                   for probas_for_POVM, n_measurements_for_POVM in zip(probas, n_measurements)]
        results = np.hstack(results)

        if warm_start:
            self.POVM_matrix = np.vstack((
                self.POVM_matrix * np.sum(self.n_measurements),
                POVM_matrix * np.sum(n_measurements),
            )) / (np.sum(self.n_measurements) + np.sum(n_measurements))
            self.results = np.hstack((self.results, results))
            self.n_measurements = np.hstack((self.n_measurements, n_measurements))
        else:
            self.POVM_matrix = POVM_matrix
            self.results = results
            self.n_measurements = np.array(n_measurements)

    def point_estimate(self, method='lin', physical=True, init='lin', max_iter=100, tol=1e-3):
        """Reconstruct a density matrix from the data obtained in the experiment

        Parameters
        ----------
        method : str, default='lin'
            Method of reconstructing the density matrix

            Possible values:
                'lin' -- linear inversion
                'mle' -- maximum likelihood estimation with Cholesky parameterization, unconstrained optimization
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

        max_iter : int, default=100 (optional)
            Number of iterations in MLE method

        tol : float, default=1e-3 (optional)
            Error tolerance in MLE method

        Returns
        -------
        reconstructed_state : Qobj
        """
        if method == 'lin':
            self.reconstructed_state = self._point_estimate_lin(physical=physical)
        elif method == 'mle':
            self.reconstructed_state = self._point_estimate_mle_chol(init=init, max_iter=max_iter, tol=tol)
        elif method == 'mle-constr':
            self.reconstructed_state = self._point_estimate_mle_chol_constr(init=init, max_iter=max_iter, tol=tol)
        elif method == 'mle-bloch':
            self.reconstructed_state = self._point_estimate_mle_bloch(physical=physical)
        else:
            raise ValueError('Invalid value for argument `method`')
        return self.reconstructed_state

    def mhmc(self, n_boot, step=0.25, burn_steps=1000, thinning=4, use_new_estimate=False, state=None):
        """Short summary.

        Parameters
        ----------
        n_boot : int
            Number of samples to be produced by MCMC.
        step : float
            Multiplier used in each step.
        burn_steps : int
            Steps for burning in.
        thinning : int
            Takes each `thinning` sample generated by MCMC.
        use_new_estimate : bool, default=False
            If False, uses the latest reconstructed state as a state to perform new tomographies on.
            If True and `state` is None, reconstruct a density matrix from the data obtained in previous experiment
            ans use it to perform new tomographies on.
            If True and `state` is not None, use `state` as a state to perform new tomographies on.
        state : Qobj or None, default=None
            If not None and `use_new_estimate` is True, use it as a state to perform new tomographies on

        Returns
        -------
        dist : np.array
            Sorted list of distances between the reconstructed state and secondary samples.
        """
        if not use_new_estimate:
            state = self.reconstructed_state
        elif state is None:
            state = self.point_estimate(method='mle', physical=True)

        target_logpdf = lambda x: -self._neg_log_likelihood_chol(x)
        dim = 4 ** self.state.n_qubits
        chain = MHMC(target_logpdf, step=step, burn_steps=burn_steps, dim=dim,
                     update_rule=normalized_update, symmetric=True)
        samples, _ = chain.sample(n_boot, thinning)
        dist = np.asarray([self.dst(_real_tril_vec_to_matrix(tril_vec), state.matrix) for tril_vec in samples])
        dist.sort()
        return dist

    def bootstrap(self, n_boot, est_method='lin', physical=True, init='lin', tol=1e-3, max_iter=100,
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
        max_iter : int, default=100 (optional)
            Number of iterations in MLE method
        tol : float, default=1e-3 (optional)
            Error tolerance in MLE method
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

        Returns
        -------
        dist : np.array
            Sorted list of distances between the reconstructed state and secondary samples.
        """
        if not use_new_estimate:
            state = self.reconstructed_state
        elif state is None:
            state = self.point_estimate(method=est_method, physical=physical, init=init, tol=tol, max_iter=max_iter)

        dist = np.empty(n_boot)
        boot_tmg = self.__class__(state, self.dst)
        for i in range(n_boot):
            boot_tmg.experiment(self.n_measurements, self.POVM_matrix)
            rho = boot_tmg.point_estimate(method=est_method, physical=physical, init=init, tol=tol, max_iter=max_iter)
            if kind == 'estim':
                dist[i] = self.dst(rho, state)
            elif kind == 'target':
                dist[i] = self.dst(rho, self.state)
            elif kind == 'triangle':
                dist[i] = self.dst(rho, state) + self.dst(state, self.state)
            else:
                raise ValueError('Invalid value for argument `kind`')
        dist.sort()
        return dist

    def _point_estimate_lin(self, physical):
        """Point estimate based on linear inversion algorithm"""
        frequencies = self.results / self.results.sum()
        POVM_matrix = np.reshape(self.POVM_matrix * self.n_measurements[:, None, None] / np.sum(self.n_measurements),
                                 (-1, self.POVM_matrix.shape[-1]))
        bloch_vec = _left_inv(POVM_matrix) @ frequencies / (2 ** self.state.n_qubits)
        rho = Qobj(bloch_vec)
        if physical:
            rho = _make_feasible(rho)
        return rho

    def _point_estimate_mle_chol(self, init, max_iter, tol):
        """Point estimate based on MLE with Cholesky parametrization"""
        if init == 'mixed':
            x0 = fully_mixed(self.state.n_qubits).matrix
        elif init == 'lin':
            x0 = self.point_estimate('lin').matrix
        else:
            raise ValueError('Invalid value for argument `init`')
        x0 = _matrix_to_real_tril_vec(x0)
        opt_res = minimize(self._neg_log_likelihood_chol, x0, method='BFGS',
                           tol=tol, options={'maxiter': max_iter})
        matrix = _real_tril_vec_to_matrix(opt_res.x)
        return Qobj(matrix / np.trace(matrix))

    def _neg_log_likelihood_chol(self, tril_vec):
        """Negative log-likelihood for MLE with Cholesky parametrization"""
        EPS = 1e-10
        matrix = _real_tril_vec_to_matrix(tril_vec)
        rho = Qobj(matrix / np.trace(matrix))
        POVM_matrix = np.reshape(self.POVM_matrix * self.n_measurements[:, None, None] / np.sum(self.n_measurements),
                                 (-1, self.POVM_matrix.shape[-1]))
        probas = POVM_matrix @ rho.bloch * (2 ** self.state.n_qubits)
        log_likelihood = np.sum(self.results * np.log(probas + EPS))
        return -log_likelihood

    def _point_estimate_mle_chol_constr(self, init, max_iter, tol):
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
        opt_res = minimize(self._neg_log_likelihood_chol, x0, constraints=constraints, method='SLSQP',
                           tol=tol, options={'maxiter': max_iter})
        matrix = _real_tril_vec_to_matrix(opt_res.x)
        return Qobj(matrix / np.trace(matrix))

    def _neg_log_likelihood_chol_constr(self, tril_vec):
        """Negative log-likelihood for constrained MLE with Cholesky parametrization"""
        EPS = 1e-10
        rho = Qobj(_real_tril_vec_to_matrix(tril_vec))
        POVM_matrix = np.reshape(self.POVM_matrix * self.n_measurements[:, None, None] / np.sum(self.n_measurements),
                                 (-1, self.POVM_matrix.shape[-1]))
        probas = POVM_matrix @ rho.bloch * (2 ** self.state.n_qubits)
        log_likelihood = np.sum(self.results * np.log(probas + EPS))
        return -log_likelihood
