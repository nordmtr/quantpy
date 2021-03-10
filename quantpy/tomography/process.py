import numpy as np
import scipy.linalg as la
import scipy.stats as sts
import itertools as it

from .state import StateTomograph
from ..geometry import hs_dst, if_dst, trace_dst
from ..routines import (
    generate_single_entries, kron,
    _real_tril_vec_to_matrix,
    _out_ptrace_oper,
    _vec2mat, _mat2vec,
    _left_inv
)
from ..stats import l2_mean, l2_variance
from ..qobj import Qobj, fully_mixed
from ..channel import Channel
from ..measurements import generate_measurement_matrix
from ..basis import Basis
from ..mhmc import MHMC


class ProcessTomograph:
    """Basic class for quantum process tomography.

    This class supports experiment simulations, different methods of reconstructing a Choi matrix
    from the data and building confidence intervals.

    Parameters
    ----------
    channel : Channel
        Quantum channel to perform a tomography on
    input_states : str or list, default='proj4'
        Set of quantum states to be used in the process tomography
    dst : str or callable, default='hs'
        Measure in a space of quantum objects

        Possible strings:
            'hs' -- Hilbert-Schmidt distance
            'trace' -- trace distance
            'if' -- infidelity

        Interface for a custom measure:
            custom_measure(A: Qobj, B: Qobj) -> float
    input_impurity : float, default=0.05
        Depolarize each input state using depolarizing channel with p = `input_impurity`
        in order to avoid biased point estimate

    Attributes
    ----------
    input_basis : Basis
        Basis of the input states
    reconstructed_channel : Channel
        The most recent estimation of a channel, if ever performed
    tomographs : list
        List of StateTomograph objects corresponding to each input state

    Methods
    -------
    bootstrap()
        Perform multiple tomography simulation
    experiment()
        Simulate a real quantum process tomography
    point_estimate()
        Reconstruct a channel from the data obtained in the experiment
    """

    def __init__(self, channel, input_states='proj4', dst='hs', input_impurity=0):
        self.channel = channel
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
        self.input_states = input_states
        self.input_impurity = input_impurity
        self.input_basis = Basis(
            _generate_input_states(input_states, input_impurity, channel.n_qubits))
        if self.input_basis.dim != 4 ** channel.n_qubits:
            raise ValueError('Input states do not constitute a basis')
        self._decomposed_single_entries = np.array([
            self.input_basis.decompose(Qobj(single_entry))
            for single_entry in generate_single_entries(2 ** channel.n_qubits)
        ])
        self._ptrace_oper = _out_ptrace_oper(channel.n_qubits)
        self._ptrace_dag_ptrace = self._ptrace_oper.T.conj() @ self._ptrace_oper

    def experiment(self, n_measurements, POVM='proj-set', warm_start=False):
        """Simulate a real quantum process tomography by performing
        quantum state tomography on each of transformed input states.

        Parameters
        ----------
        n_measurements : int
            Number of measurements to perform in the tomography
        POVM : str or numpy 2-D array, default='proj'
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
            If True, do not overwrite the previous experiment results, add all results to those
            of the previous run
        """
        if not warm_start:
            self.tomographs = []
            for input_state in self.input_basis.elements:
                output_state_true = self.channel.transform(input_state)
                tmg = StateTomograph(output_state_true)
                self.tomographs.append(tmg)
        for tmg in self.tomographs:
            tmg.experiment(n_measurements, POVM, warm_start=warm_start)

    def point_estimate(self, method='lifp', cptp=True, n_iter=1000, tol=1e-10,
                       states_est_method='lin', states_physical=True, states_init='lin'):
        """Reconstruct a Choi matrix from the data obtained in the experiment.

        Parameters
        ----------
        method : str, default='lifp'
            Method of reconstructing the Choi matrix

            Possible values:
                'lifp' -- linear inversion
                'pgdb' -- projected gradient descent (CPTP only)
                'states' -- reconstruction of the Choi matrix using a basis of reconstructed
                quantum states

        cptp : bool, default=True
            If True, return a projection onto CPTP space.

        states_est_method : str, default='lin' (optional)
            Method of reconstructing of every output state (only if method='states')

            Possible values:
                'lin' -- linear inversion
                'mle' -- maximum likelihood estimation with Cholesky parametrization,
                unconstrained optimization
                'mle-constr' -- same as 'mle', but optimization is constrained
                'mle-bloch' -- maximum likelihood estimation with Bloch parametrization,
                               constrained optimization (works only for 1-qubit systems)

        states_physical : bool, default=True (optional)
           For 'states' method defines if the point estimates of the quantum states should be
           physical

        states_init : str, default='lin' (optional)
           For 'states' method with MLE sets an initial point for gradient descent

           Possible values:
               'lin' -- uses linear inversion point estimate as initial guess
               'mixed' -- uses fully mixed state as initial guess

        Returns
        -------
        reconstructed_channel : Channel
        """
        dim = 2 ** self.channel.n_qubits
        self._lifp_oper = []
        self._bloch_oper = []
        POVM_matrix = np.reshape(
            self.tomographs[0].POVM_matrix * self.tomographs[0].n_measurements[:, None, None]
            / np.sum(self.tomographs[0].n_measurements),
            (-1, self.tomographs[0].POVM_matrix.shape[-1])
        )
        for inp_state, POVM_bloch in it.product(self.input_basis.elements, POVM_matrix):
            row = _mat2vec(np.kron(inp_state.matrix, Qobj(POVM_bloch).matrix.T))
            self._lifp_oper.append(row)
            self._bloch_oper.append(np.kron(inp_state.T.bloch, POVM_bloch))

        self._lifp_oper = np.array(self._lifp_oper)
        self._bloch_oper = np.array(self._bloch_oper) * dim ** 2
        self._lifp_oper_inv = _left_inv(self._lifp_oper)
        self._bloch_oper_inv = _left_inv(self._bloch_oper)

        self._unnorm_results = np.hstack([stmg.results for stmg in self.tomographs])

        if method == 'lifp':
            return self._point_estimate_lifp(cptp=cptp)
        elif method == 'pgdb':
            return self._point_estimate_pgdb(n_iter=n_iter, tol=tol)
        elif method == 'states':
            return self._point_estimate_states(
                cptp=cptp, method=states_est_method, physical=states_physical, init=states_init)
        else:
            raise ValueError('Incorrect value for argument `method`')

    def gamma_interval(self, n_points):
        """Use gamma distribution approximation to obtain confidence interval.

        Parameters
        ----------
        n_points : int
            Number of distances to get.

        Returns
        -------
        dist : np.array
            Sorted list of distances between the reconstructed state and secondary samples.
        CLs : np.array
            List of corresponding confidence levels.
        """
        if not hasattr(self, '_lifp_oper_inv'):
            _ = self.point_estimate('lifp')
        n_measurements = np.hstack([tmg.n_measurements for tmg in self.tomographs])
        long_n_measurements = n_measurements.astype(object)
        measurement_ratios = long_n_measurements / long_n_measurements.sum() * len(self.tomographs)
        raw_results = np.vstack([tmg.raw_results for tmg in self.tomographs])
        frequencies = raw_results / n_measurements[:, None]
        means = l2_mean(frequencies, long_n_measurements)
        mean = np.sum(means * measurement_ratios ** 2)
        variances = l2_variance(frequencies, long_n_measurements)
        variance = np.sum(variances * measurement_ratios ** 4)
        scale = variance / mean
        shape = mean / scale
        gamma = sts.gamma(a=shape, scale=scale)
        CLs = np.linspace(0.001, 0.999, n_points)
        dim = 2 ** self.channel.n_qubits
        if self.dst == hs_dst:
            alpha = 1 / np.sqrt(2)
        elif self.dst == trace_dst:
            alpha = dim / 2
        else:
            raise NotImplementedError()
        dist = np.sqrt(gamma.ppf(CLs)) * alpha * np.linalg.norm(self._lifp_oper_inv, ord=2)
        return dist, CLs

    def mhmc(self, n_points, step=0.01, burn_steps=1000, thinning=1, warm_start=False,
             method='lifp',
             states_est_method='lin', states_physical=True, states_init='lin',
             use_new_estimate=False,
             channel=None, verbose=False, return_samples=False):
        """Use Metropolis-Hastings Monte Carlo algorithm to obtain samples from likelihood
        distribution.
        Count the distances between these samples and point estimate.

        Parameters
        ----------
        n_points : int
            Number of samples to be produced by MCMC.
        step : float
            Multiplier used in each step.
        burn_steps : int
            Steps for burning in.
        thinning : int
            Takes each `thinning` sample generated by MCMC.
        warm_start : bool
            If True, the warmed up chain is used.
        use_new_estimate : bool, default=False
            If False, uses the latest reconstructed channel as a channel to perform new
            tomographies on.
            If True and `channel` is None, reconstruct a density matrix from the data obtained in
            previous experiment
            ans use it to perform new tomographies on.
            If True and `channel` is not None, use `channel` as a channel to perform new
            tomographies on.
        channel : Channel or None, default=None
            If not None and `use_new_estimate` is True, use it as a channel to perform new
            tomographies on
        verbose : bool
            If True, shows progress.
        return_samples : bool
            If `return_matrices` returns additionally list of MHMC samples.

        Returns
        -------
        dist : np.array
            Sorted list of distances between the reconstructed channel and secondary samples.
        CLs : np.array
            List of corresponding confidence levels.
        acceptance_rate : float
            Fraction of accepted samples.
        samples : list of numpy 2D arrays
            If `return_samples` returns list of MHMC samples.
        """
        if not use_new_estimate:
            channel = self.reconstructed_channel
        elif channel is None:
            channel = self.point_estimate(method, states_est_method=states_est_method,
                                          states_physical=states_physical, states_init=states_init)

        target_logpdf = lambda x: -self._nll(x)
        dim = 16 ** self.channel.n_qubits
        if not (warm_start and hasattr(self, 'chain')):
            x_init = _mat2vec(channel.choi.matrix)
            self.chain = MHMC(target_logpdf, step=step, burn_steps=burn_steps, dim=dim,
                              update_rule=self._cptp_update_rule, symmetric=True, x_init=x_init)
        samples, acceptance_rate = self.chain.sample(n_points, thinning, verbose=verbose)
        dist = np.asarray(
            [self.dst(_vec2mat(choi_vec), channel.choi.matrix) for choi_vec in samples])
        dist.sort()
        CLs = np.linspace(0, 1, len(dist))
        if return_samples:
            matrices = [_vec2mat(choi_vec) for choi_vec in samples]
            return dist, CLs, acceptance_rate, matrices
        return dist, CLs, acceptance_rate

    def bootstrap(self, n_points, method='lifp', cptp=True, tol=1e-10, use_new_estimate=False,
                  channel=None, states_est_method='lin', states_physical=True, states_init='lin'):
        """Perform multiple tomography simulation on the preferred channel with the same
        measurements number
        and POVM matrix, as in the preceding experiment. Count the distances to the bootstrapped
        Choi matrices.

        Parameters
        ----------
        n_points : int
            Number of experiments to perform
        method : str, default='lifp'
            Method of reconstructing the Choi matrix
            See :ref:`point_estimate` for detailed documentation
        states_est_method : str, default='lin'
            Method of reconstructing the density matrix for each output state
            See :ref:`point_estimate` for detailed documentation
        states_physical : bool, default=True (optional)
            See :ref:`point_estimate` for detailed documentation
        states_init : str, default='lin' (optional)
            See :ref:`point_estimate` for detailed documentation
        use_new_estimate : bool, default=False
            If False, uses the latest reconstructed channel as a channel to perform new
            tomographies on.
            If True and `channel` is None, reconstruct a density matrix from the data obtained in
            previous experiment
            ans use it to perform new tomographies on.
            If True and `channel` is not None, use `channel` as a channel to perform new
            tomographies on.
        channel : Qobj or None, default=None
            If not None and `use_new_estimate` is True, use it as a channel to perform new
            tomographies on
        cptp : bool, default=True
            If True, all bootstrap samples are projected onto CPTP space

        Returns
        -------
        dist : np.array
            Sorted list of distances between the reconstructed channel and secondary samples.
        CLs : np.array
            List of corresponding confidence levels.
        """
        if not use_new_estimate:
            channel = self.reconstructed_channel
        elif channel is None:
            channel = self.point_estimate(method=method, states_physical=states_physical,
                                          states_init=states_init, cptp=cptp)

        dist = np.empty(n_points)
        boot_tmg = self.__class__(channel, self.input_states, self.dst, self.input_impurity)
        for i in range(n_points):
            boot_tmg.experiment(self.tomographs[0].n_measurements,
                                POVM=self.tomographs[0].POVM_matrix)
            estim_channel = boot_tmg.point_estimate(method=method, states_physical=states_physical,
                                                    states_init=states_init, cptp=cptp)
            dist[i] = self.dst(estim_channel.choi, channel.choi)
        dist.sort()
        CLs = np.linspace(0, 1, len(dist))
        return dist, CLs

    def holder_interval(self, n_points, interval='gamma', method='lin', method_boot='lin',
                        physical=True, init='lin', tol=1e-3, max_iter=100, step=0.01,
                        burn_steps=1000, thinning=1, wang_mode='coarse'):
        """Conducts `n_iter` experiments, constructs confidence intervals for each,
        computes confidence level that corresponds to the distance between
        the target state and the point estimate and returns a sorted list of these levels.

        Parameters
        ----------
        n_points : int
            Number of distances to get.
        interval : str
            Method of constructing the interval.

            Possible values:
                'gamma' -- theoretical interval based on approximation with gamma distribution
                'boot' -- bootstrapping from the point estimate
                'mhmc' -- Metropolis-Hastings Monte Carlo
                'sugiyama' -- 1306.4191 interval
                'wang' -- 1808.09988 interval
        method : str
            Method of reconstructing the density matrix

            Possible values:
                'lin' -- linear inversion
                'mle' -- maximum likelihood estimation with Cholesky parameterization,
                unconstrained optimization
                'mle-constr' -- same as 'mle', but optimization is constrained
                'mle-bloch' -- maximum likelihood estimation with Bloch parametrization,
                               constrained optimization (works only for 1-qubit systems)

        method_boot : str
            Method of reconstructing the bootstrapped samples. See method() documentation for the
            details.

        physical : bool (optional)
            For methods 'lin' and 'mle' reconstructed matrix may not lie in the physical domain.
            If True, set negative eigenvalues to zeros and divide the matrix by its trace.

        init : str (optional)
            Methods using maximum likelihood estimation require the starting point for gradient
            descent.

            Possible values:
                'lin' -- uses linear inversion point estimate as initial guess.
                'mixed' -- uses fully mixed state as initial guess.

        max_iter : int (optional)
            Number of iterations in MLE method.
        tol : float (optional)
            Error tolerance in MLE method.
        step : float
            Multiplier used in each step.
        burn_steps : int
            Steps for burning in.
        thinning : int
            Takes each `thinning` sample generated by MCMC.
        wang_mode : str
            Mode for Wang et al. method

        Returns
        -------
        dist : np.array
            Sorted list of distances between the reconstructed channel and secondary samples.
        CLs : np.array
            List of corresponding confidence levels.
        """
        if interval == 'gamma':
            state_results = [tmg.gamma_interval(n_points) for tmg in self.tomographs]
        elif interval == 'mhmc':
            state_results = [tmg.mhmc(n_points, step, burn_steps, thinning) for tmg in
                             self.tomographs]
        elif interval == 'boot':
            state_results = [
                tmg.bootstrap(n_points, method_boot, physical=physical, init=init, tol=tol,
                              max_iter=max_iter)
                for tmg in self.tomographs
            ]
        elif interval == 'sugiyama':
            state_results = [tmg.sugiyama_interval(n_points) for tmg in self.tomographs]
        elif interval == 'wang':
            state_results = [tmg.wang_interval(n_points, mode=wang_mode) for tmg in self.tomographs]
        else:
            raise ValueError('Incorrect value for argument `interval`.')

        state_deltas = np.asarray([state_result[0] for state_result in state_results])
        CLs = state_results[0][1]

        coef = np.abs(np.einsum('ij,ik->jk', self._decomposed_single_entries,
                                self._decomposed_single_entries.conj()))
        state_deltas_composition = np.einsum('ik,jk->ijk', state_deltas, state_deltas)
        dist = np.sqrt(np.einsum('ijk,ij->k', state_deltas_composition, coef))
        return dist, CLs

    def cptp_projection(self, channel, n_iter=1000, tol=1e-12):
        """Implementation of an iterative CPTP projection subroutine"""
        choi_vec = _mat2vec(channel.choi.matrix)
        cptp_choi_vec = self._cptp_projection_vec(choi_vec, n_iter, tol)
        return Channel(_vec2mat(cptp_choi_vec))

    def _cptp_projection_vec(self, choi_vec, n_iter=1000, tol=1e-12):
        """Implementation of an iterative CPTP projection subroutine"""
        x = choi_vec
        p = 0
        q = 0
        y = 0
        for i in range(n_iter):
            stop_criterion_value = 0
            y_diff = self.tp_projection(Channel(_vec2mat(x + p)), vectorized=True) - y
            y += y_diff
            x_diff = self.cp_projection(Channel(_vec2mat(y + q)), vectorized=True) - x
            x += x_diff
            stop_criterion_value += 2 * (
                        np.abs(np.sum(y_diff.T.conj() * q)) + np.abs(np.sum(x_diff.T.conj() * p)))
            p_diff = x - y
            p += p_diff
            q_diff = y - x
            q += q_diff
            stop_criterion_value += la.norm(p_diff) ** 2 + la.norm(q_diff) ** 2
            if stop_criterion_value < tol:
                break
        return x

    def tp_projection(self, channel, vectorized=False):
        """Projection of a channel onto TP space"""
        dim = 2 ** channel.n_qubits
        choi_vec = _mat2vec(channel.choi.matrix)
        tp_choi_vec = choi_vec + (
                self._ptrace_oper.T.conj() @ _mat2vec(np.eye(dim))
                - self._ptrace_dag_ptrace @ choi_vec
        ) / dim
        if vectorized:
            return tp_choi_vec
        return Channel(_vec2mat(tp_choi_vec))

    def cp_projection(self, channel, vectorized=False):
        """Projection of a channel onto CP space"""
        EPS = 1e-12
        v, U = la.eigh(channel.choi.matrix)
        V = np.diag(np.maximum(EPS, v))
        cp_choi_matrix = U @ V @ U.T.conj()
        if vectorized:
            return _mat2vec(cp_choi_matrix)
        return Channel(cp_choi_matrix)

    def _cptp_update_rule(self, x_t, delta, step):
        noncptp_x_prime = x_t + step * delta
        return self._cptp_projection_vec(noncptp_x_prime)

    def _point_estimate_lifp(self, cptp):
        self.frequencies = np.hstack(
            [stmg.results / stmg.results.sum() for stmg in self.tomographs])
        self.reconstructed_channel = Channel(_vec2mat(self._lifp_oper_inv @ self.frequencies))
        if cptp:
            self.reconstructed_channel = self.cptp_projection(self.reconstructed_channel)
        return self.reconstructed_channel

    def _point_estimate_pgdb(self, n_iter, tol=1e-10):
        choi_vec = _mat2vec(fully_mixed(self.channel.n_qubits * 2).matrix)
        mu = 1.5 / (4 ** self.channel.n_qubits)
        gamma = 0.3
        for i in range(n_iter):
            probas = self._lifp_oper @ choi_vec
            grad = -self._lifp_oper.T.conj() @ (self._unnorm_results / probas)
            D = self._cptp_projection_vec(choi_vec - grad / mu) - choi_vec
            alpha = 1
            while self._nll(choi_vec + alpha * D) - self._nll(choi_vec) > gamma * alpha * np.dot(D,
                                                                                                 grad):
                alpha /= 2
            new_choi_vec = choi_vec + alpha * D
            if self._nll(choi_vec) - self._nll(new_choi_vec) > tol:
                break
            choi_vec = new_choi_vec

        self.reconstructed_channel = Channel(_vec2mat(choi_vec))
        return self.reconstructed_channel

    def _nll(self, choi_vec):
        EPS = 1e-12
        probas = self._lifp_oper @ choi_vec
        log_likelihood = np.sum(self._unnorm_results * np.log(probas + EPS))
        return -log_likelihood

    def _point_estimate_states(self, cptp, method, physical, init):
        output_states = [tmg.point_estimate(method, physical, init) for tmg in self.tomographs]
        output_basis = Basis(output_states)
        choi_matrix = Qobj(np.zeros((output_basis.dim, output_basis.dim)))
        for decomposed_single_entry in self._decomposed_single_entries:
            single_entry = self.input_basis.compose(decomposed_single_entry)
            transformed_single_entry = output_basis.compose(decomposed_single_entry)
            choi_matrix += kron(single_entry, transformed_single_entry)
        self.reconstructed_channel = Channel(choi_matrix)
        if cptp and not self.reconstructed_channel.is_cptp(verbose=False):
            self.reconstructed_channel = self.cptp_projection(self.reconstructed_channel)
        return self.reconstructed_channel


def _generate_input_states(type, input_impurity, n_qubits):
    """Generate input states to use in quantum process tomography"""
    if isinstance(type, list):
        return type
    input_states_list = []
    for input_state_bloch in np.squeeze(generate_measurement_matrix(type, n_qubits)):
        input_state = Qobj(input_state_bloch)
        input_state /= input_state.trace()
        input_states_list.append(input_state)
    return input_states_list


def _tp_constraint(tril_vec):
    choi = Qobj(_real_tril_vec_to_matrix(tril_vec))
    rho_in = _vec2mat(_out_ptrace_oper(choi.n_qubits // 2) @ _mat2vec(choi.matrix))
    return hs_dst(rho_in, np.eye(2 ** (choi.n_qubits // 2)))
