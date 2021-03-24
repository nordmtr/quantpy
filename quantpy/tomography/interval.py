import numpy as np
import scipy.stats as sts
import polytope as pc
import math
import pypoman

from enum import Enum, auto
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d

from ..geometry import hs_dst, trace_dst, if_dst
from ..polytope import compute_polytope_volume, find_max_distance_to_polytope
from ..qobj import Qobj
from ..routines import (
    _left_inv, _vec2mat, _mat2vec,
    _matrix_to_real_tril_vec, _real_tril_vec_to_matrix,
)
from ..stats import l2_mean, l2_variance
from ..mhmc import MHMC, normalized_update
from .state import _make_feasible


class ConfidenceInterval(ABC):
    """Functor for obtaining confidence intervals."""

    def __init__(self, tmg, *args, **kwargs):
        """
        Parameters
        ----------
        tmg : StateTomograph or ProcessTomograph
            Object with tomography results
        """
        self.tmg = tmg
        if hasattr(tmg, 'state'):
            self.mode = Mode.STATE
        elif hasattr(tmg, 'channel'):
            self.mode = Mode.CHANNEL
        else:
            raise ValueError()
        for name, value in kwargs.items():
            setattr(self, name, value)

    @abstractmethod
    def __call__(self):
        """Return confidence interval.

        Returns
        -------
        dist : np.array
            Sorted list of distances between the reconstructed state and secondary samples.
        CLs : np.array
            List of corresponding confidence levels.
        """
        pass


# noinspection PyProtectedMember
class GammaInterval(ConfidenceInterval):
    def __init__(self, tmg, n_points=1000, max_confidence=0.999):
        """Use gamma distribution approximation to obtain confidence interval.

        Parameters
        ----------
        tmg : StateTomograph or ProcessTomograph
            Object with tomography results
        n_points : int
            Number of distances to get.
        max_confidence : float
            Maximum confidence level
        """
        kwargs = locals()
        for key in ('self', 'tmg'):
            kwargs.pop(key)
        super().__init__(tmg, **kwargs)

    def __call__(self):
        if self.mode == Mode.STATE:
            long_n_measurements = self.tmg.n_measurements.astype(object)
            measurement_ratios = long_n_measurements / long_n_measurements.sum()
            frequencies = self.tmg.raw_results / self.tmg.n_measurements[:, None]
        else:
            if not hasattr(self, '_lifp_oper_inv'):
                _ = self.tmg.point_estimate('lifp')
            n_measurements = np.hstack([tmg.n_measurements for tmg in self.tmg.tomographs])
            long_n_measurements = n_measurements.astype(object)
            measurement_ratios = long_n_measurements / long_n_measurements.sum() * len(
                self.tmg.tomographs)
            raw_results = np.vstack([tmg.raw_results for tmg in self.tmg.tomographs])
            frequencies = raw_results / n_measurements[:, None]
        means = l2_mean(frequencies, long_n_measurements)
        mean = np.sum(means * measurement_ratios ** 2)
        variances = l2_variance(frequencies, long_n_measurements)
        variance = np.sum(variances * measurement_ratios ** 4)
        scale = variance / mean
        shape = mean / scale
        gamma = sts.gamma(a=shape, scale=scale)
        # noinspection PyUnresolvedReferences
        CLs = np.linspace(0.001, self.max_confidence, self.n_points)
        if self.mode == Mode.STATE:
            dim = 2 ** self.tmg.state.n_qubits
            if self.tmg.dst == hs_dst:
                alpha = np.sqrt(dim / 2)
            elif self.tmg.dst == trace_dst:
                alpha = dim / 2
            else:
                raise NotImplementedError()
            povm_matrix = np.reshape(self.tmg.povm_matrix * self.tmg.n_measurements[:, None, None]
                                     / np.sum(self.tmg.n_measurements),
                                     (-1, self.tmg.povm_matrix.shape[-1]))
            A = _left_inv(povm_matrix) / dim
            dist = np.sqrt(gamma.ppf(CLs)) * alpha * np.linalg.norm(A, ord=2)
        else:
            dim = 2 ** self.tmg.channel.n_qubits
            if self.tmg.dst == hs_dst:
                alpha = 1 / np.sqrt(2)
            elif self.tmg.dst == trace_dst:
                alpha = dim / 2
            else:
                raise NotImplementedError()
            dist = np.sqrt(gamma.ppf(CLs)) * alpha * np.linalg.norm(self.tmg._lifp_oper_inv, ord=2)
        return dist, CLs


class SugiyamaInterval(ConfidenceInterval):
    def __init__(self, tmg, n_points=1000, max_confidence=0.999):
        """Construct a confidence interval based on Hoeffding inequality as in work 1306.4191 of
        Sugiyama et al.

        Parameters
        ----------
        tmg : StateTomograph
            Object with tomography results
        n_points : int
            Number of distances to get.
        max_confidence : float
            Maximum confidence level
        """
        kwargs = locals()
        for key in ('self', 'tmg'):
            kwargs.pop(key)
        super().__init__(tmg, **kwargs)

    def __call__(self):
        if self.mode == Mode.CHANNEL:
            raise NotImplementedError("Sugiyama interval works only for state tomography")
        EPS = 1e-15
        dim = 2 ** self.tmg.state.n_qubits
        dist_dummy = np.linspace(0, 1, 100000)
        povm_matrix = np.reshape(self.tmg.povm_matrix, (-1, self.tmg.povm_matrix.shape[-1])) * dim
        povm_matrix /= np.sqrt(2 * dim)
        inversed_povm = _left_inv(povm_matrix).reshape(
            (-1, self.tmg.povm_matrix.shape[0], self.tmg.povm_matrix.shape[1]))
        measurement_ratios = self.tmg.n_measurements.sum() / self.tmg.n_measurements
        c_alpha = np.sum((np.max(inversed_povm, axis=-1) - np.min(inversed_povm, axis=-1)) ** 2
                         * measurement_ratios[None, :], axis=-1) + EPS
        if self.tmg.dst == hs_dst:
            b = 8 / (dim ** 2 - 1)
        elif self.tmg.dst == trace_dst:
            b = 16 / (dim ** 2 - 1) / dim
        elif self.tmg.dst == if_dst:
            b = 4 / (dim ** 2 - 1) / dim
        else:
            raise NotImplementedError("Unsupported distance")
        CLs_dummy = 1 - 2 * np.sum(
            np.exp(-b * dist_dummy[:, None] ** 2
                   * np.sum(self.tmg.n_measurements) / c_alpha[None, :]), axis=1)
        cl_to_dist = interp1d(CLs_dummy, dist_dummy)
        CLs = np.linspace(0, self.max_confidence, self.n_points)
        dist = cl_to_dist(CLs)
        return dist, CLs


class WangInterval(ConfidenceInterval):
    def __init__(self, tmg, n_points=1000, method='bbox', max_confidence=0.999):
        """Construct a confidence interval based on Clopper-Pearson interval as in work
        1808.09988 of Wang et al.

        Parameters
        ----------
        tmg : StateTomograph
            Object with tomography results
        n_points : int
            Number of distances to get.
        method : str
            Method of calculating the radius of the circumscribed sphere.
        max_confidence : float
            Maximum confidence level
        """
        kwargs = locals()
        for key in ('self', 'tmg'):
            kwargs.pop(key)
        super().__init__(tmg, **kwargs)

    def __call__(self):
        if self.mode == Mode.CHANNEL:
            raise NotImplementedError("Wang interval works only for state tomography")
        EPS = 1e-15
        rho = self.tmg.point_estimate('lin', physical=False)
        dim = 2 ** self.tmg.state.n_qubits
        bloch_dim = dim ** 2 - 1

        frequencies = np.clip(self.tmg.raw_results / self.tmg.n_measurements[:, None], EPS, 1 - EPS)

        povm_matrix = (np.reshape(self.tmg.povm_matrix * self.tmg.n_measurements[:, None, None]
                                  / np.sum(self.tmg.n_measurements),
                                  (-1, self.tmg.povm_matrix.shape[-1]))
                       * dim * self.tmg.povm_matrix.shape[0])

        max_delta = self._count_delta(self.max_confidence, frequencies)

        if self.method == 'coarse':
            A = _left_inv(povm_matrix)
            prob_dim = povm_matrix.shape[0]
            coef1 = (np.linalg.norm(A, ord=2)
                     * (self.tmg.povm_matrix.shape[1] - 1)
                     * np.sqrt(prob_dim))
            coef2 = np.linalg.norm(A, ord=2) ** 2 * prob_dim
            max_dist = max(max_delta * coef1, np.sqrt(max_delta * coef2))
            dist_dummy = np.linspace(0, max_dist, self.n_points)
            deltas = np.maximum(dist_dummy / coef1, dist_dummy ** 2 / coef2)
        else:
            deltas = np.linspace(0, max_delta, self.n_points)
            dist_dummy = []
            A = np.ascontiguousarray(povm_matrix[:, 1:])
            for delta in deltas:
                b = np.clip(np.hstack(frequencies) + delta, EPS, 1 - EPS) - povm_matrix[:, 0] / dim
                if self.method == 'exact':
                    vertices = pypoman.compute_polytope_vertices(A, b)
                    vertex_states = [_make_feasible(Qobj(vertex)) for vertex in vertices]
                    if vertices:
                        radius = max([self.tmg.dst(vertex_state, rho) for vertex_state in
                                      vertex_states])
                    else:
                        radius = 0
                elif self.method == 'bbox':
                    lb, ub = pc.Polytope(A, b).bounding_box
                    volume = np.prod(ub - lb)
                    radius = ((volume * math.gamma(bloch_dim / 2 + 1)) ** (1 / bloch_dim)
                              / math.sqrt(math.pi))
                elif self.method == 'approx':
                    volume = compute_polytope_volume(pc.Polytope(A, b))
                    radius = ((volume * math.gamma(bloch_dim / 2 + 1)) ** (1 / bloch_dim)
                              / math.sqrt(math.pi))
                elif self.method == 'hit_and_run':
                    rho_bloch = rho.bloch[1:]
                    radius = find_max_distance_to_polytope(A, b, rho_bloch, rho_bloch)
                else:
                    raise ValueError("Invalid value for argument `mode`.")
                dist_dummy.append(radius)

        CLs_dummy = []
        for delta in deltas:
            CLs_dummy.append(self._count_confidence(delta, frequencies))
        cl_to_dist = interp1d(CLs_dummy, dist_dummy)
        CLs = np.linspace(0, self.max_confidence, self.n_points)
        dist = cl_to_dist(CLs)
        return dist, CLs

    def _count_confidence(self, delta, frequencies):
        EPS = 1e-15
        freq_plus_delta = np.clip(frequencies + delta, EPS, 1 - EPS)
        KL_divergence = (frequencies * np.log(frequencies / freq_plus_delta)
                         + (1 - frequencies) * np.log(
                    (1 - frequencies) / (1 - freq_plus_delta)))
        KL_divergence = np.where(freq_plus_delta < 1 - EPS, KL_divergence, np.inf)
        epsilons = np.exp(-self.tmg.n_measurements[:, None] * KL_divergence)
        epsilons = np.where(np.abs(frequencies - 1) < 2 * EPS, 0, epsilons)
        return 1 - np.sum(epsilons)

    def _count_delta(self, confidence_threshold, frequencies):
        delta = 1e-10
        while True:
            confidence = self._count_confidence(delta, frequencies)
            if confidence > confidence_threshold:
                return delta
            delta *= 2


# noinspection PyProtectedMember,PyProtectedMember
class HolderInterval(ConfidenceInterval):
    def __init__(self, tmg, n_points=1000, kind='gamma', max_confidence=0.999,
                 method='lin', method_boot='lin', physical=True, init='lin', tol=1e-3,
                 max_iter=100, step=0.01, burn_steps=1000, thinning=1, wang_method='bbox'):
        """Conducts `n_points` experiments, constructs confidence intervals for each,
        computes confidence level that corresponds to the distance between
        the target state and the point estimate and returns a sorted list of these levels.

        Parameters
        ----------
        tmg : ProcessTomograph
            Object with tomography results
        n_points : int
            Number of distances to get.
        kind : str
            Method of constructing the interval.

            Possible values:
                'gamma' -- theoretical interval based on approximation with gamma distribution
                'boot' -- bootstrapping from the point estimate
                'mhmc' -- Metropolis-Hastings Monte Carlo
                'sugiyama' -- 1306.4191 interval
                'wang' -- 1808.09988 interval
        max_confidence : float
            Maximum confidence level for 'gamma', 'wang' and 'sugiyama' methods.
        method : str
            Method of reconstructing the density matrix of bootstrap samples

            Possible values:
                'lin' -- linear inversion
                'mle' -- maximum likelihood estimation with Cholesky parameterization,
                unconstrained optimization
                'mle-constr' -- same as 'mle', but optimization is constrained
                'mle-bloch' -- maximum likelihood estimation with Bloch parametrization,
                               constrained optimization (works only for 1-qubit systems)

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
        wang_method : str
            Method for Wang method

        Returns
        -------
        dist : np.array
            Sorted list of distances between the reconstructed channel and secondary samples.
        CLs : np.array
            List of corresponding confidence levels.
        """
        kwargs = locals()
        for key in ('self', 'tmg'):
            kwargs.pop(key)
        super().__init__(tmg, **kwargs)

    def __call__(self):
        if self.mode == Mode.STATE:
            raise NotImplementedError("Holder interval works only for process tomography")
        if self.kind == 'gamma':
            state_results = [GammaInterval(tmg, self.n_points, self.max_confidence)()
                             for tmg in self.tmg.tomographs]
        elif self.kind == 'mhmc':
            state_results = [
                MHMCStateInterval(tmg, self.n_points, self.step, self.burn_steps, self.thinning)()
                for tmg in self.tmg.tomographs
            ]
        elif self.kind == 'bootstrap':
            state_results = [
                BootstrapStateInterval(tmg, self.n_points, self.method, physical=self.physical,
                                       init=self.init, tol=self.tol, max_iter=self.max_iter)()
                for tmg in self.tmg.tomographs
            ]
        elif self.kind == 'sugiyama':
            state_results = [SugiyamaInterval(tmg, self.n_points, self.max_confidence)()
                             for tmg in self.tmg.tomographs]
        elif self.kind == 'wang':
            state_results = [
                WangInterval(tmg, self.n_points, self.wang_method, self.max_confidence)()
                for tmg in self.tmg.tomographs
            ]
        else:
            raise ValueError('Incorrect value for argument `kind`.')

        state_deltas = np.asarray([state_result[0] for state_result in state_results])
        CLs = state_results[0][1]

        coef = np.abs(np.einsum('ij,ik->jk', self.tmg._decomposed_single_entries,
                                self.tmg._decomposed_single_entries.conj()))
        state_deltas_composition = np.einsum('ik,jk->ijk', state_deltas, state_deltas)
        dist = np.sqrt(np.einsum('ijk,ij->k', state_deltas_composition, coef))
        return dist, CLs


class BootstrapStateInterval(ConfidenceInterval):
    def __init__(self, tmg, n_points=1000, method='lin', physical=True,
                 init='lin', tol=1e-3, max_iter=100, state=None):
        """Perform multiple tomography simulation on the preferred state with the same
        measurements number
        and POVM matrix, as in the preceding experiment. Count the distances to the
        bootstrapped
        states.

        Parameters
        ----------
        tmg : StateTomograph
            Object with tomography results
        n_points : int
            Number of experiments to perform
        method : str, default='lin'
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
        state : Qobj or None, default=None
            If not None, use it as a state to perform new tomographies on.
            Otherwise use the reconstructed state from tmg.
        """
        kwargs = locals()
        for key in ('self', 'tmg'):
            kwargs.pop(key)
        super().__init__(tmg, **kwargs)

    def __call__(self):
        if self.mode == Mode.CHANNEL:
            raise NotImplementedError("This interval works only for state tomography")
        if self.state is None:
            if hasattr(self.tmg, 'reconstructed_state'):
                self.state = self.tmg.reconstructed_state
            else:
                self.state = self.tmg.point_estimate(method=self.method, physical=self.physical,
                                                     init=self.init, tol=self.tol,
                                                     max_iter=self.max_iter)

        dist = np.empty(self.n_points)
        boot_tmg = self.tmg.__class__(self.state, self.tmg.dst)
        for i in range(self.n_points):
            boot_tmg.experiment(self.tmg.n_measurements, self.tmg.povm_matrix)
            rho = boot_tmg.point_estimate(method=self.method, physical=self.physical,
                                          init=self.init, tol=self.tol, max_iter=self.max_iter)
            dist[i] = self.tmg.dst(rho, self.state)
        dist.sort()
        CLs = np.linspace(0, 1, len(dist))
        return dist, CLs


class BootstrapProcessInterval(ConfidenceInterval):
    def __init__(self, tmg, n_points=1000, method='lifp', cptp=True, tol=1e-10, channel=None,
                 states_est_method='lin', states_physical=True, states_init='lin'):
        """Perform multiple tomography simulation on the preferred channel with the same
        measurements number
        and POVM matrix, as in the preceding experiment. Count the distances to the
        bootstrapped
        Choi matrices.

        Parameters
        ----------
        tmg : ProcessTomograph
            Object with tomography results
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
        channel : Channel or None, default=None
            If not None, use it as a channel to perform new tomographies on.
            Otherwise use the reconstructed channel from tmg.
        cptp : bool, default=True
            If True, all bootstrap samples are projected onto CPTP space
        """
        kwargs = locals()
        for key in ('self', 'tmg'):
            kwargs.pop(key)
        super().__init__(tmg, **kwargs)

    def __call__(self):
        if self.mode == Mode.STATE:
            raise NotImplementedError("This interval works only for process tomography")
        if self.channel is None:
            if hasattr(self.tmg, 'reconstructed_channel'):
                self.channel = self.tmg.reconstructed_channel
            else:
                self.channel = self.tmg.point_estimate(
                    method=self.method, states_physical=self.states_physical,
                    states_init=self.states_init, cptp=self.cptp
                )

        dist = np.empty(self.n_points)
        boot_tmg = self.tmg.__class__(self.channel, self.tmg.input_states, self.tmg.dst)
        for i in range(self.n_points):
            boot_tmg.experiment(self.tmg.tomographs[0].n_measurements,
                                povm=self.tmg.tomographs[0].povm_matrix)
            estim_channel = boot_tmg.point_estimate(
                method=self.method, states_physical=self.states_physical,
                states_init=self.states_init, cptp=self.cptp
            )
            dist[i] = self.tmg.dst(estim_channel.choi, self.channel.choi)
        dist.sort()
        CLs = np.linspace(0, 1, len(dist))
        return dist, CLs


# noinspection PyProtectedMember
class MHMCStateInterval(ConfidenceInterval):
    def __init__(self, tmg, n_points=1000, step=0.01, burn_steps=1000, thinning=1,
                 warm_start=False, use_new_estimate=False, state=None, verbose=False):
        """Use Metropolis-Hastings Monte Carlo algorithm to obtain samples from likelihood
        distribution.
        Count the distances between these samples and point estimate.

        Parameters
        ----------
        tmg : StateTomograph
            Object with tomography results
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
            If False, uses the latest reconstructed state as a state to perform new tomographies on.
            If True and `state` is None, reconstruct a density matrix from the data obtained in
            previous experiment
            ans use it to perform new tomographies on.
            If True and `state` is not None, use `state` as a state to perform new tomographies on.
        state : Qobj or None, default=None
            If not None and `use_new_estimate` is True, use it as a state to perform new
            tomographies on.
        verbose: bool
            If True, shows progress.
        """
        kwargs = locals()
        for key in ('self', 'tmg'):
            kwargs.pop(key)
        super().__init__(tmg, **kwargs)

    # noinspection PyTypeChecker
    def __call__(self):
        if self.mode == Mode.CHANNEL:
            raise NotImplementedError("This interval works only for state tomography")
        if not self.use_new_estimate:
            self.state = self.tmg.reconstructed_state
        elif self.state is None:
            self.state = self.tmg.point_estimate(method='mle', physical=True)

        dim = 4 ** self.tmg.state.n_qubits
        if not (self.warm_start and hasattr(self, 'chain')):
            x_init = _matrix_to_real_tril_vec(self.state.matrix)
            self.chain = MHMC(
                lambda x: -self.tmg._nll(x), step=self.step, burn_steps=self.burn_steps,
                dim=dim, update_rule=normalized_update, symmetric=True, x_init=x_init
            )
        samples, acceptance_rate = self.chain.sample(self.n_points, self.thinning,
                                                     verbose=self.verbose)
        dist = np.asarray([self.tmg.dst(_real_tril_vec_to_matrix(tril_vec), self.state.matrix)
                           for tril_vec in samples])
        dist.sort()
        CLs = np.linspace(0, 1, len(dist))
        return dist, CLs, acceptance_rate


# noinspection PyProtectedMember,PyProtectedMember
class MHMCProcessInterval(ConfidenceInterval):
    def __init__(self, tmg, n_points=1000, step=0.01, burn_steps=1000, thinning=1, warm_start=False,
                 method='lifp', states_est_method='lin', states_physical=True, states_init='lin',
                 use_new_estimate=False, channel=None, verbose=False, return_samples=False):
        """Use Metropolis-Hastings Monte Carlo algorithm to obtain samples from likelihood
        distribution.
        Count the distances between these samples and point estimate.

        Parameters
        ----------
        tmg : ProcessTomograph
            Object with tomography results
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
        """
        kwargs = locals()
        for key in ('self', 'tmg'):
            kwargs.pop(key)
        super().__init__(tmg, **kwargs)

    def __call__(self):
        if self.mode == Mode.STATE:
            raise NotImplementedError("This interval works only for process tomography")
        if not self.use_new_estimate:
            self.channel = self.tmg.reconstructed_channel
        elif self.channel is None:
            self.channel = self.tmg.point_estimate(
                self.method, states_est_method=self.states_est_method,
                states_physical=self.states_physical, states_init=self.states_init
            )

        # noinspection PyPep8,PyPep8
        target_logpdf = lambda x: -self.tmg._nll(x)
        dim = 16 ** self.tmg.channel.n_qubits
        if not (self.warm_start and hasattr(self, 'chain')):
            x_init = _mat2vec(self.channel.choi.matrix)
            # noinspection PyTypeChecker
            self.chain = MHMC(target_logpdf, step=self.step, burn_steps=self.burn_steps, dim=dim,
                              update_rule=self.tmg._cptp_update_rule, symmetric=True, x_init=x_init)
        samples, acceptance_rate = self.chain.sample(self.n_points, self.thinning,
                                                     verbose=self.verbose)
        dist = np.asarray(
            [self.tmg.dst(_vec2mat(choi_vec), self.channel.choi.matrix) for choi_vec in samples])
        dist.sort()
        CLs = np.linspace(0, 1, len(dist))
        if self.return_samples:
            matrices = [_vec2mat(choi_vec) for choi_vec in samples]
            return dist, CLs, acceptance_rate, matrices
        return dist, CLs, acceptance_rate


class Mode(Enum):
    STATE = auto()
    CHANNEL = auto()
