import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

from ..geometry import hs_dst, if_dst, trace_dst
from ..measurements import generate_measurement_matrix
from ..qobj import Qobj, fully_mixed
from ..routines import _left_inv, _matrix_to_real_tril_vec, _real_tril_vec_to_matrix


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
    povm_matrix : numpy 2-D array
        Numpy array with shape (*, 4^n_qubits), representing the measurement matrix.
        Rows are bloch vectors that sum into unity
    reconstructed_state : Qobj
        The most recent estimation of a density matrix, if ever performed
    flat_results : numpy 1-D array
        Results of the simulated experiment.
        flat_results[i] is the number of outcomes corresponding to POVM_matrix[i,:]

    Methods
    -------
    bootstrap()
        Perform multiple tomography simulation
    experiment()
        Simulate a real quantum state tomography
    point_estimate()
        Reconstruct a density matrix from the data obtained in the experiment
    """

    def __init__(self, state, dst="hs"):
        self.state = state
        if isinstance(dst, str):
            if dst == "hs":
                self.dst = hs_dst
            elif dst == "trace":
                self.dst = trace_dst
            elif dst == "if":
                self.dst = if_dst
            else:
                raise ValueError("Invalid value for argument `dst`")
        else:
            self.dst = dst

        self._results = None

    def experiment(self, n_measurements, povm="proj-set", warm_start=False):
        """Simulate a real quantum state tomography.

        Parameters
        ----------
        n_measurements : int or array-like
            Number of measurements to perform in the tomography
        povm : str or numpy 2-D array, default='proj'
            A single string or a numpy array to construct a POVM matrix.

            Possible strings:
                'proj' -- random orthogonal projective measurement, 6^n_qubits rows
                'proj-set' -- true orthogonal projective measurement, set of POVMs
                'sic' -- SIC POVM for 1-qubit systems and its tensor products for higher
                dimensions, 4^n_qubits rows

            Possible numpy arrays:
                2-D array with shape (*, 4) -- interpreted as POVM matrix for 1 qubit,
                construct a POVM matrix for the whole system from tensor products of rows of this
                matrix
                3-D array with shape (*, *, 4) -- same, but set of POVMs
                2-D array with shape (*, 4^n_qubits) -- returns this matrix without any changes
                3-D array with shape (*, *, 4^n_qubits) -- same, but set of POVMs

            See :ref:`generate_measurement_matrix` for more detailed documentation

        warm_start : bool, default=False
            If True, do not overwrite the previous experiment flat_results, add all flat_results to those
            of the previous run
        """
        povm_matrix = generate_measurement_matrix(povm, self.state.n_qubits)
        number_of_povms = povm_matrix.shape[0]

        if np.issubdtype(type(n_measurements), np.integer):
            n_measurements = np.ones(number_of_povms) * n_measurements
        elif len(n_measurements) != number_of_povms:
            raise ValueError("Wrong length for argument `n_measurements`")

        probas = np.einsum("ijk,k->ij", povm_matrix, self.state.bloch) * (2**self.state.n_qubits)
        probas = np.clip(probas, 0, 1)
        results = [
            np.random.multinomial(n_measurements_for_povm, probas_for_povm)
            for probas_for_povm, n_measurements_for_povm in zip(probas, n_measurements)
        ]

        if warm_start:
            self.povm_matrix = np.vstack(
                (
                    self.povm_matrix * np.sum(self.n_measurements),
                    povm_matrix * np.sum(n_measurements),
                )
            ) / (np.sum(self.n_measurements) + np.sum(n_measurements))
            self.n_measurements = np.hstack((self.n_measurements, n_measurements))
            self.results = np.vstack((self.results, results))
        else:
            self.povm_matrix = povm_matrix
            self.results = np.asarray(results)
            self.n_measurements = np.asarray(n_measurements)

    @property
    def flat_results(self):
        return self.results.flatten()

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, results):
        self._results = results
        self.n_measurements = results.sum(-1)

    def point_estimate(self, method="lin", physical=True, init="lin", max_iter=100, tol=1e-3):
        """Reconstruct a density matrix from the data obtained in the experiment

        Parameters
        ----------
        method : str, default='lin'
            Method of reconstructing the density matrix

            Possible values:
                'lin' -- linear inversion
                'mle' -- maximum likelihood estimation with Cholesky parameterization,
                unconstrained optimization
                'mle-constr' -- same as 'mle', but optimization is constrained
                'mle-bloch' -- maximum likelihood estimation with Bloch parametrization,
                               constrained optimization (works only for 1-qubit systems)

        physical : bool, default=True (optional)
            For methods 'lin' and 'mle' reconstructed matrix may not lie in the physical domain.
            If True, set negative eigenvalues to zeros and divide the matrix by its trace.

        init : str, default='lin' (optional)
            Methods using maximum likelihood estimation require the starting point for gradient
            descent.

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
        if method == "lin":
            self.reconstructed_state = self._point_estimate_lin(physical=physical)
        elif method == "mle":
            self.reconstructed_state = self._point_estimate_mle_chol(init=init, max_iter=max_iter, tol=tol)
        elif method == "mle-constr":
            self.reconstructed_state = self._point_estimate_mle_chol_constr(init=init, max_iter=max_iter, tol=tol)
        else:
            raise ValueError("Invalid value for argument `method`")
        return self.reconstructed_state

    def _point_estimate_lin(self, physical):
        """Point estimate based on linear inversion algorithm"""
        frequencies = self.flat_results / self.flat_results.sum()
        povm_matrix = np.reshape(
            self.povm_matrix * self.n_measurements[:, None, None] / np.sum(self.n_measurements),
            (-1, self.povm_matrix.shape[-1]),
        )
        bloch_vec = _left_inv(povm_matrix) @ frequencies / (2**self.state.n_qubits)
        rho = Qobj(bloch_vec)
        if physical:
            rho = _make_feasible(rho)
        return rho

    def _point_estimate_mle_chol(self, init, max_iter, tol):
        """Point estimate based on MLE with Cholesky parametrization"""
        if init == "mixed":
            x0 = fully_mixed(self.state.n_qubits).matrix
        elif init == "lin":
            x0 = self.point_estimate("lin").matrix
        else:
            raise ValueError("Invalid value for argument `init`")
        x0 = _matrix_to_real_tril_vec(x0)
        opt_res = minimize(self._nll, x0, method="BFGS", tol=tol, options={"maxiter": max_iter})
        matrix = _real_tril_vec_to_matrix(opt_res.x)
        return Qobj(matrix / np.trace(matrix))

    def _nll(self, tril_vec):
        """Negative log-likelihood for MLE with Cholesky parametrization"""
        EPS = 1e-10
        matrix = _real_tril_vec_to_matrix(tril_vec)
        rho = Qobj(matrix / np.trace(matrix))
        povm_matrix = np.reshape(
            self.povm_matrix * self.n_measurements[:, None, None] / np.sum(self.n_measurements),
            (-1, self.povm_matrix.shape[-1]),
        )
        probas = povm_matrix @ rho.bloch * (2**self.state.n_qubits)
        frequencies = self.flat_results / sum(self.n_measurements)
        log_likelihood = np.sum(frequencies * np.log(probas + EPS))
        return -log_likelihood

    def _point_estimate_mle_chol_constr(self, init, max_iter, tol):
        """Point estimate based on constrained MLE with Cholesky parametrization"""
        constraints = [
            {"type": "eq", "fun": _is_unit_trace},
        ]
        if init == "mixed":
            x0 = fully_mixed(self.state.n_qubits).matrix
        elif init == "lin":
            x0 = self.point_estimate("lin").matrix
        else:
            raise ValueError("Invalid value for argument `init`")
        x0 = _matrix_to_real_tril_vec(x0)
        # noinspection PyTypeChecker
        opt_res = minimize(
            self._nll,
            x0,
            constraints=constraints,
            method="SLSQP",
            tol=tol,
            options={"maxiter": max_iter},
        )
        matrix = _real_tril_vec_to_matrix(opt_res.x)
        return Qobj(matrix / np.trace(matrix))


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
