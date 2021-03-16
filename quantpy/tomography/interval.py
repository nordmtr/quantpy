import numpy as np
import scipy.stats as sts
import polytope as pc
import math
import pypoman

from enum import Enum, auto
from abc import ABC, abstractmethod

from ..geometry import hs_dst, trace_dst
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
    def __init__(self, tmg):
        self.tmg = tmg
        if hasattr(tmg, 'state'):
            self.mode = Mode.STATE
        elif hasattr(tmg, 'channel'):
            self.mode = Mode.CHANNEL
        else:
            raise ValueError()

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class GammaInterval(ConfidenceInterval):
    def __call__(self, n_points=1000):
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
        CLs = np.linspace(0.001, 0.999, n_points)
        if self.mode == Mode.STATE:
            dim = 2 ** self.tmg.state.n_qubits
            if self.tmg.dst == hs_dst:
                alpha = np.sqrt(dim / 2)
            elif self.tmg.dst == trace_dst:
                alpha = dim / 2
            else:
                raise NotImplementedError()
            POVM_matrix = np.reshape(self.tmg.POVM_matrix * self.tmg.n_measurements[:, None, None]
                                     / np.sum(self.tmg.n_measurements),
                                     (-1, self.tmg.POVM_matrix.shape[-1]))
            A = _left_inv(POVM_matrix) / dim
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
    def __call__(self, n_points=1000, dist=None):
        """Construct a confidence interval based on Hoeffding inequality as in work 1306.4191 of
        Sugiyama et al.

        Parameters
        ----------
        n_points : int
            Number of distances to get.
        dist : array-like
            Sorted list of distances between the reconstructed state and secondary samples
            where to calculate confidence levels.

        Returns
        -------
        dist : np.array
            Sorted list of distances between the reconstructed state and secondary samples.
        CLs : np.array
            List of corresponding confidence levels.
        """
        if self.mode == Mode.CHANNEL:
            raise NotImplementedError("Sugiyama interval works only for state tomography")
        EPS = 1e-15
        if dist is None:
            dist = np.linspace(0, 1, n_points)
        POVM_matrix = (np.reshape(self.tmg.POVM_matrix, (-1, self.tmg.POVM_matrix.shape[-1]))
                       * 2 ** self.tmg.state.n_qubits)
        inversed_POVM = _left_inv(POVM_matrix).reshape(
            (-1, self.tmg.POVM_matrix.shape[0], self.tmg.POVM_matrix.shape[1]))
        measurement_ratios = self.tmg.n_measurements.sum() / self.tmg.n_measurements
        c_alpha = np.sum((np.max(inversed_POVM, axis=-1) - np.min(inversed_POVM, axis=-1)) ** 2
                         * measurement_ratios[None, :], axis=-1) + EPS
        b = 4 / POVM_matrix.shape[1]
        CLs = 1 - 2 * np.sum(
            np.exp(-b * dist[:, None] ** 2 * np.sum(self.tmg.n_measurements) / c_alpha[None, :]),
            axis=1)
        return dist, CLs


class WangInterval(ConfidenceInterval):
    def __call__(self, n_points, mode='bbox'):
        """Construct a confidence interval based on Clopper-Pearson interval as in work
        1808.09988 of Wang et al.

        Parameters
        ----------
        n_points : int
            Number of distances to get.
        mode : str
            Method of calculating the radius of the circumscribed sphere.

        Returns
        -------
        dist : np.array
            Sorted list of distances between the reconstructed state and secondary samples.
        CLs : np.array
            List of corresponding confidence levels.
        """
        if self.mode == Mode.CHANNEL:
            raise NotImplementedError("Wang interval works only for state tomography")
        EPS = 1e-15
        rho = self.tmg.point_estimate('lin', physical=False)
        dim = 2 ** self.tmg.state.n_qubits
        bloch_dim = dim ** 2 - 1

        frequencies = np.clip(self.tmg.raw_results / self.tmg.n_measurements[:, None], EPS, 1 - EPS)

        POVM_matrix = (np.reshape(self.tmg.POVM_matrix * self.tmg.n_measurements[:, None, None]
                                  / np.sum(self.tmg.n_measurements),
                                  (-1, self.tmg.POVM_matrix.shape[-1]))
                       * dim * self.tmg.POVM_matrix.shape[0])

        if mode == 'coarse':
            A = _left_inv(POVM_matrix)
            prob_dim = POVM_matrix.shape[0]
            coef1 = (np.linalg.norm(A, ord=2)
                     * (self.tmg.POVM_matrix.shape[1] - 1)
                     * np.sqrt(prob_dim))
            coef2 = np.linalg.norm(A, ord=2) ** 2 * prob_dim
            dist = np.linspace(0, 1, n_points)
            deltas = np.maximum(dist / coef1, dist ** 2 / coef2)
        else:
            deltas = np.linspace(0, 0.2, n_points)
            dist = []
            A = np.ascontiguousarray(POVM_matrix[:, 1:])
            for delta in deltas:
                b = np.clip(np.hstack(frequencies) + delta, EPS, 1 - EPS) - POVM_matrix[:, 0] / dim
                if mode == 'exact':
                    vertices = pypoman.compute_polytope_vertices(A, b)
                    vertex_states = [_make_feasible(Qobj(vertex)) for vertex in vertices]
                    if vertices:
                        radius = max([self.tmg.dst(vertex_state, rho) for vertex_state in
                                      vertex_states])
                    else:
                        radius = 0
                elif mode == 'bbox':
                    lb, ub = pc.Polytope(A, b).bounding_box
                    volume = np.prod(ub - lb)
                    radius = ((volume * math.gamma(bloch_dim / 2 + 1)) ** (1 / bloch_dim)
                              / math.sqrt(math.pi))
                elif mode == 'approx':
                    volume = compute_polytope_volume(pc.Polytope(A, b))
                    radius = ((volume * math.gamma(bloch_dim / 2 + 1)) ** (1 / bloch_dim)
                              / math.sqrt(math.pi))
                elif mode == 'hit_and_run':
                    rho_bloch = rho.bloch[1:]
                    radius = find_max_distance_to_polytope(A, b, rho_bloch, rho_bloch)
                else:
                    raise ValueError("Invalid value for argument `mode`.")
                dist.append(radius)

        CLs = []
        for delta in deltas:
            freq_plus_delta = np.clip(frequencies + delta, EPS, 1 - EPS)
            KL_divergence = (frequencies * np.log(frequencies / freq_plus_delta)
                             + (1 - frequencies) * np.log(
                        (1 - frequencies) / (1 - freq_plus_delta)))
            KL_divergence = np.where(freq_plus_delta < 1 - EPS, KL_divergence, np.inf)
            epsilons = np.exp(-self.tmg.n_measurements[:, None] * KL_divergence)
            epsilons = np.where(np.abs(frequencies - 1) < 2 * EPS, 0, epsilons)

            CLs.append(1 - np.sum(epsilons))
        return np.asarray(dist), np.asarray(CLs)


class HolderInterval(ConfidenceInterval):
    def __call__(self, n_points, interval='gamma', method='lin', method_boot='lin',
                 physical=True, init='lin', tol=1e-3, max_iter=100, step=0.01,
                 burn_steps=1000, thinning=1, wang_mode='coarse'):
        """Conducts `n_points` experiments, constructs confidence intervals for each,
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
        if self.mode == Mode.STATE:
            raise NotImplementedError("Holder interval works only for process tomography")
        if interval == 'gamma':
            state_results = [GammaInterval(tmg)(n_points) for tmg in self.tmg.tomographs]
        elif interval == 'mhmc':
            state_results = [MHMCStateInterval(tmg)(n_points, step, burn_steps, thinning)
                             for tmg in self.tmg.tomographs]
        elif interval == 'boot':
            state_results = [
                BootstrapStateInterval(tmg)(n_points, method_boot, physical=physical, init=init,
                                            tol=tol, max_iter=max_iter)
                for tmg in self.tmg.tomographs
            ]
        elif interval == 'sugiyama':
            state_results = [SugiyamaInterval(tmg)(n_points) for tmg in self.tmg.tomographs]
        elif interval == 'wang':
            state_results = [WangInterval(tmg)(n_points, mode=wang_mode)
                             for tmg in self.tmg.tomographs]
        else:
            raise ValueError('Incorrect value for argument `interval`.')

        state_deltas = np.asarray([state_result[0] for state_result in state_results])
        CLs = state_results[0][1]

        coef = np.abs(np.einsum('ij,ik->jk', self.tmg._decomposed_single_entries,
                                self.tmg._decomposed_single_entries.conj()))
        state_deltas_composition = np.einsum('ik,jk->ijk', state_deltas, state_deltas)
        dist = np.sqrt(np.einsum('ijk,ij->k', state_deltas_composition, coef))
        return dist, CLs


class BootstrapStateInterval(ConfidenceInterval):
    def __call__(self, n_points, method='lin', physical=True, init='lin', tol=1e-3, max_iter=100,
                 use_new_estimate=False, state=None):
        """Perform multiple tomography simulation on the preferred state with the same
        measurements number
        and POVM matrix, as in the preceding experiment. Count the distances to the bootstrapped
        states.

        Parameters
        ----------
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
        use_new_estimate : bool, default=False
            If False, uses the latest reconstructed state as a state to perform new tomographies on.
            If True and `state` is None, reconstruct a density matrix from the data obtained in
            previous experiment
            ans use it to perform new tomographies on.
            If True and `state` is not None, use `state` as a state to perform new tomographies on.
        state : Qobj or None, default=None
            If not None and `use_new_estimate` is True, use it as a state to perform new
            tomographies on

        Returns
        -------
        dist : np.array
            Sorted list of distances between the reconstructed state and secondary samples.
        CLs : np.array
            List of corresponding confidence levels.
        """
        if self.mode == Mode.CHANNEL:
            raise NotImplementedError("This interval works only for state tomography")
        if not use_new_estimate:
            state = self.tmg.reconstructed_state
        elif state is None:
            state = self.tmg.point_estimate(method=method, physical=physical, init=init, tol=tol,
                                            max_iter=max_iter)

        dist = np.empty(n_points)
        boot_tmg = self.tmg.__class__(state, self.tmg.dst)
        for i in range(n_points):
            boot_tmg.experiment(self.tmg.n_measurements, self.tmg.POVM_matrix)
            rho = boot_tmg.point_estimate(method=method, physical=physical, init=init, tol=tol,
                                          max_iter=max_iter)
            dist[i] = self.tmg.dst(rho, state)
        dist.sort()
        CLs = np.linspace(0, 1, len(dist))
        return dist, CLs


class BootstrapProcessInterval(ConfidenceInterval):
    def __call__(self, n_points, method='lifp', cptp=True, tol=1e-10, use_new_estimate=False,
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
        if self.mode == Mode.STATE:
            raise NotImplementedError("This interval works only for process tomography")
        if not use_new_estimate:
            channel = self.tmg.reconstructed_channel
        elif channel is None:
            channel = self.tmg.point_estimate(method=method, states_physical=states_physical,
                                              states_init=states_init, cptp=cptp)

        dist = np.empty(n_points)
        boot_tmg = self.tmg.__class__(channel, self.tmg.input_states, self.tmg.dst)
        for i in range(n_points):
            boot_tmg.experiment(self.tmg.tomographs[0].n_measurements,
                                POVM=self.tmg.tomographs[0].POVM_matrix)
            estim_channel = boot_tmg.point_estimate(method=method, states_physical=states_physical,
                                                    states_init=states_init, cptp=cptp)
            dist[i] = self.tmg.dst(estim_channel.choi, channel.choi)
        dist.sort()
        CLs = np.linspace(0, 1, len(dist))
        return dist, CLs


class MHMCStateInterval(ConfidenceInterval):
    def __call__(self, n_points, step=0.01, burn_steps=1000, thinning=1, warm_start=False,
                 use_new_estimate=False, state=None, verbose=False):
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

        Returns
        -------
        dist : np.array
            Sorted list of distances between the reconstructed state and secondary samples.
        CLs : np.array
            List of corresponding confidence levels.
        acceptance_rate : float
            Fraction of samples accepted by the Metropolis-Hastings procedure
        """
        if self.mode == Mode.CHANNEL:
            raise NotImplementedError("This interval works only for state tomography")
        if not use_new_estimate:
            state = self.tmg.reconstructed_state
        elif state is None:
            state = self.tmg.point_estimate(method='mle', physical=True)

        dim = 4 ** self.tmg.state.n_qubits
        if not (warm_start and hasattr(self, 'chain')):
            x_init = _matrix_to_real_tril_vec(state.matrix)
            self.chain = MHMC(lambda x: -self.tmg._nll(x), step=step, burn_steps=burn_steps, dim=dim,
                              update_rule=normalized_update, symmetric=True, x_init=x_init)
        samples, acceptance_rate = self.chain.sample(n_points, thinning, verbose=verbose)
        dist = np.asarray([self.tmg.dst(_real_tril_vec_to_matrix(tril_vec), state.matrix)
                           for tril_vec in samples])
        dist.sort()
        CLs = np.linspace(0, 1, len(dist))
        return dist, CLs, acceptance_rate


class MHMCProcessInterval(ConfidenceInterval):
    def __call__(self, n_points, step=0.01, burn_steps=1000, thinning=1, warm_start=False,
                 method='lifp', states_est_method='lin', states_physical=True, states_init='lin',
                 use_new_estimate=False, channel=None, verbose=False, return_samples=False):
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
        if self.mode == Mode.STATE:
            raise NotImplementedError("This interval works only for process tomography")
        if not use_new_estimate:
            channel = self.tmg.reconstructed_channel
        elif channel is None:
            channel = self.tmg.point_estimate(method, states_est_method=states_est_method,
                                              states_physical=states_physical,
                                              states_init=states_init)

        target_logpdf = lambda x: -self.tmg._nll(x)
        dim = 16 ** self.tmg.channel.n_qubits
        if not (warm_start and hasattr(self, 'chain')):
            x_init = _mat2vec(channel.choi.matrix)
            self.chain = MHMC(target_logpdf, step=step, burn_steps=burn_steps, dim=dim,
                              update_rule=self.tmg._cptp_update_rule, symmetric=True, x_init=x_init)
        samples, acceptance_rate = self.chain.sample(n_points, thinning, verbose=verbose)
        dist = np.asarray(
            [self.tmg.dst(_vec2mat(choi_vec), channel.choi.matrix) for choi_vec in samples])
        dist.sort()
        CLs = np.linspace(0, 1, len(dist))
        if return_samples:
            matrices = [_vec2mat(choi_vec) for choi_vec in samples]
            return dist, CLs, acceptance_rate, matrices
        return dist, CLs, acceptance_rate


class Mode(Enum):
    STATE = auto()
    CHANNEL = auto()
