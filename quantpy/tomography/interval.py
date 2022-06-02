from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np
import scipy.stats as sts
from cvxopt import matrix, solvers
from einops import rearrange, repeat
from scipy.interpolate import interp1d

from ..geometry import hs_dst, if_dst, trace_dst
from ..mhmc import MHMC, normalized_update
from ..routines import _left_inv, _mat2vec, _matrix_to_real_tril_vec, _real_tril_vec_to_matrix, _vec2mat
from ..stats import l2_mean, l2_variance
from .polytopes.utils import count_confidence, count_delta

solvers.options["show_progress"] = False


class ConfidenceInterval(ABC):
    """Functor for obtaining confidence intervals."""

    EPS = 1e-15

    def __init__(self, tmg, **kwargs):
        """
        Parameters
        ----------
        tmg : StateTomograph or ProcessTomograph
            Object with tomography flat_results
        """
        self.tmg = tmg
        if hasattr(tmg, "state"):
            self.mode = Mode.STATE
        elif hasattr(tmg, "channel"):
            self.mode = Mode.CHANNEL
        else:
            raise ValueError()
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __call__(self, conf_levels=None):
        """Return confidence interval.

        Returns
        -------
        conf_levels : np.array
            List of confidence levels.
        """
        if conf_levels is None:
            conf_levels = np.linspace(1e-3, 1 - 1e-3, 1000)
        if not hasattr(self, "cl_to_dist"):
            self.setup()
        return self.cl_to_dist(conf_levels), conf_levels

    @abstractmethod
    def setup(self):
        """Configure confidence intervals based on several points and interpolation."""


class MomentInterval(ConfidenceInterval):
    def __init__(self, tmg, distr_type="gamma"):
        """Use moments to obtain confidence interval.

        Parameters
        ----------
        tmg : StateTomograph or ProcessTomograph
            Object with tomography flat_results
        """
        kwargs = _pop_hidden_keys(locals())
        super().__init__(tmg, **kwargs)

    def setup(self):
        if self.mode == Mode.STATE:
            dim = 2**self.tmg.state.n_qubits
            n_measurements = self.tmg.n_measurements
            frequencies = self.tmg.results / self.tmg.n_measurements[:, None]
            # reshape, invert, reshape back
            inv_matrix = _left_inv(rearrange(self.tmg.povm_matrix, "m p d -> (m p) d")) / dim
            inv_matrix = rearrange(inv_matrix, "d (m p) -> d m p", m=frequencies.shape[0])
        else:
            dim = 4**self.tmg.channel.n_qubits
            n_measurements = self.tmg.tomographs[0].n_measurements
            frequencies = np.vstack([tmg.results / n_measurements[:, None] for tmg in self.tmg.tomographs])
            povm_matrix = rearrange(self.tmg.tomographs[0].povm_matrix, "m p d -> (m p) d")
            states_matrix = np.asarray([rho.T.bloch for rho in self.tmg.input_basis.elements])
            channel_matrix = np.einsum("s d, p i -> s p d i", states_matrix, povm_matrix)
            # reshape, invert, reshape back
            inv_matrix = _left_inv(rearrange(channel_matrix, "s p d i -> (s p) (d i)")) / dim
            inv_matrix = rearrange(inv_matrix, "d (m p) -> d m p", m=frequencies.shape[0])
        weights_tensor = np.einsum("aij,akl->ijkl", inv_matrix, inv_matrix)
        mean = l2_mean(frequencies, n_measurements[0], weights_tensor)
        variance = l2_variance(frequencies, n_measurements[0], weights_tensor)
        if self.distr_type == "norm":
            std = np.sqrt(variance)
            distr = sts.norm(loc=mean, scale=std)
        elif self.distr_type == "gamma":
            scale = variance / mean
            shape = mean / scale
            distr = sts.gamma(a=shape, scale=scale)
        elif self.distr_type == "exp":
            distr = sts.expon(scale=mean)
        else:
            raise NotImplementedError(f"Unsupported distribution type {self.distr_type}")

        if self.tmg.dst == hs_dst:
            alpha = np.sqrt(dim / 2)
        elif self.tmg.dst == trace_dst:
            alpha = dim / 2
        else:
            raise NotImplementedError()

        self.cl_to_dist = lambda cl: np.sqrt(distr.ppf(cl)) * alpha


class MomentFidelityStateInterval(MomentInterval):
    def __init__(self, tmg, distr_type="gamma", target_state=None):
        self.target_state = target_state
        super().__init__(tmg, distr_type=distr_type)

    def __call__(self, conf_levels=None):
        if conf_levels is None:
            conf_levels = np.linspace(1e-3, 1 - 1e-3, 1000)
        if not hasattr(self, "cl_to_dist_max"):
            self.setup()
        return (self.cl_to_dist_min(conf_levels), self.cl_to_dist_max(conf_levels)), conf_levels

    def setup(self):
        super().setup()

        if not hasattr(self.tmg, "reconstructed_state"):
            _ = self.tmg.point_estimate(physical=False)

        if self.target_state is None:
            self.target_state = self.tmg.reconstructed_state
        dim = 2**self.tmg.state.n_qubits
        conf_levels = np.concatenate((np.arange(1e-7, 0.8, 0.01), np.linspace(0.8, 1 - 1e-7, 200)))
        dist_list = self.cl_to_dist(conf_levels)

        c = matrix(self.target_state.bloch)
        A = matrix([1.0] + [0] * (dim**2 - 1), size=(1, dim**2))
        b = matrix([1 / dim])
        G = [matrix(np.vstack((np.zeros(dim**2), -np.eye(dim**2))))]
        h = [matrix([0] + list(-self.tmg.reconstructed_state.bloch))]
        alpha = np.sqrt(2 / dim)

        dist_min = []
        dist_max = []
        for dist in dist_list:
            h[0][0] = dist * alpha
            sol = solvers.socp(c, Gq=G, hq=h, A=A, b=b)
            if not sol["primal objective"]:
                dist_min.append(1)
            else:
                dist_min.append(sol["primal objective"] * dim)
            sol = solvers.socp(-c, Gq=G, hq=h, A=A, b=b)
            if not sol["primal objective"]:
                dist_max.append(1)
            else:
                dist_max.append(-sol["primal objective"] * dim)

        self.cl_to_dist_max = interp1d(conf_levels, dist_max)
        self.cl_to_dist_min = interp1d(conf_levels, dist_min)


class MomentFidelityProcessInterval(MomentInterval):
    def __init__(self, tmg, distr_type="gamma", target_process=None):
        self.target_process = target_process
        super().__init__(tmg, distr_type=distr_type)

    def __call__(self, conf_levels=None):
        if conf_levels is None:
            conf_levels = np.linspace(1e-3, 1 - 1e-3, 1000)
        if not hasattr(self, "cl_to_dist_max"):
            self.setup()
        return (self.cl_to_dist_min(conf_levels), self.cl_to_dist_max(conf_levels)), conf_levels

    def setup(self):
        super().setup()

        if not hasattr(self.tmg, "reconstructed_channel"):
            _ = self.tmg.point_estimate(cptp=False)

        if self.target_process is None:
            self.target_process = self.tmg.reconstructed_channel

        dim_in = dim_out = 2**self.tmg.channel.n_qubits
        dim = dim_in * dim_out
        trivial_indices = list(range(0, dim**2, dim_out**2))

        conf_levels = np.concatenate((np.arange(1e-7, 0.8, 0.01), np.linspace(0.8, 1 - 1e-7, 200)))
        dist_list = self.cl_to_dist(conf_levels)

        # TODO: double-check the correctness
        c = matrix(self.target_process.choi.bloch)
        A = matrix(np.eye(dim**2)[trivial_indices])
        b = matrix([1 / dim_in] + [0] * (dim_in**2 - 1))
        G = [matrix(np.vstack((np.zeros(dim**2), -np.eye(dim**2))))]
        h = [matrix([0] + list(-self.tmg.reconstructed_channel.choi.bloch))]
        alpha = np.sqrt(2 / dim)

        dist_min = []
        dist_max = []
        for dist in dist_list:
            h[0][0] = dist * alpha
            sol = solvers.socp(c, Gq=G, hq=h, A=A, b=b)
            if not sol["primal objective"]:
                dist_min.append(1)
            else:
                dist_min.append(sol["primal objective"])
            sol = solvers.socp(-c, Gq=G, hq=h, A=A, b=b)
            if not sol["primal objective"]:
                dist_max.append(1)
            else:
                dist_max.append(-sol["primal objective"])

        self.cl_to_dist_max = interp1d(conf_levels, dist_max)
        self.cl_to_dist_min = interp1d(conf_levels, dist_min)


class SugiyamaInterval(ConfidenceInterval):
    def __init__(self, tmg, n_points=1000, max_confidence=0.999):
        """Construct a confidence interval based on Hoeffding inequality as in work 1306.4191 of
        Sugiyama et al.

        Parameters
        ----------
        tmg : StateTomograph
            Object with tomography flat_results
        n_points : int
            Number of distances to get.
        max_confidence : float
            Maximum confidence level
        """
        kwargs = _pop_hidden_keys(locals())
        super().__init__(tmg, **kwargs)

    def setup(self):
        if self.mode == Mode.CHANNEL:
            raise NotImplementedError("Sugiyama interval works only for state tomography")
        dim = 2**self.tmg.state.n_qubits
        dist = np.linspace(0, 1, self.n_points)
        povm_matrix = np.reshape(self.tmg.povm_matrix, (-1, self.tmg.povm_matrix.shape[-1])) * dim
        povm_matrix /= np.sqrt(2 * dim)
        inversed_povm = _left_inv(povm_matrix).reshape(
            (-1, self.tmg.povm_matrix.shape[0], self.tmg.povm_matrix.shape[1])
        )
        measurement_ratios = self.tmg.n_measurements.sum() / self.tmg.n_measurements
        c_alpha = (
            np.sum(
                (np.max(inversed_povm, axis=-1) - np.min(inversed_povm, axis=-1)) ** 2 * measurement_ratios[None, :],
                axis=-1,
            )
            + self.EPS
        )
        if self.tmg.dst == hs_dst:
            b = 8 / (dim**2 - 1)
        elif self.tmg.dst == trace_dst:
            b = 16 / (dim**2 - 1) / dim
        elif self.tmg.dst == if_dst:
            b = 4 / (dim**2 - 1) / dim
        else:
            raise NotImplementedError("Unsupported distance")
        conf_levels = 1 - 2 * np.sum(
            np.exp(-b * dist[:, None] ** 2 * np.sum(self.tmg.n_measurements) / c_alpha[None, :]),
            axis=1,
        )
        self.cl_to_dist = interp1d(conf_levels, dist)


class PolytopeStateInterval(ConfidenceInterval):
    def __init__(self, tmg, n_points=1000, target_state=None):
        """Construct a confidence interval based on linear optimization in a polytope as in work 2109.04734 of
        Kiktenko et al.

        Parameters
        ----------
        tmg : StateTomograph
            Object with tomography flat_results
        n_points : int
            Number of distances to get.
        target_state : qp.Qobj
            If specified, calculates fidelity w.r.t. this state
        """
        kwargs = _pop_hidden_keys(locals())
        super().__init__(tmg, **kwargs)

    def __call__(self, conf_levels=None):
        if conf_levels is None:
            conf_levels = np.linspace(1e-3, 1 - 1e-3, 1000)
        if not hasattr(self, "cl_to_dist_max"):
            self.setup()
        return (self.cl_to_dist_min(conf_levels), self.cl_to_dist_max(conf_levels)), conf_levels

    def setup(self):
        if self.mode == Mode.CHANNEL:
            raise NotImplementedError("This interval works only for state tomography")

        if self.target_state is None:
            self.target_state = self.tmg.state

        dim = 2**self.tmg.state.n_qubits
        frequencies = np.clip(self.tmg.results / self.tmg.n_measurements[:, None], self.EPS, 1 - self.EPS)

        povm_matrix = (
            np.reshape(
                self.tmg.povm_matrix * self.tmg.n_measurements[:, None, None] / np.sum(self.tmg.n_measurements),
                (-1, self.tmg.povm_matrix.shape[-1]),
            )
            * self.tmg.povm_matrix.shape[0]
        )
        A = np.ascontiguousarray(povm_matrix[:, 1:]) * dim
        c = matrix(self.target_state.bloch[1:])

        max_delta = count_delta(1 - 1e-7, frequencies, self.tmg.n_measurements)
        min_delta = count_delta(0, frequencies, self.tmg.n_measurements)
        deltas = np.linspace(min_delta, max_delta, self.n_points)

        dist_max = []
        dist_min = []
        for delta in deltas:
            b = np.clip(np.hstack(frequencies) + delta, self.EPS, 1 - self.EPS) - povm_matrix[:, 0]
            G, h = matrix(A), matrix(b)
            sol = solvers.lp(c, G, h)
            if not sol["primal objective"]:
                dist_min.append(1)
            else:
                dist_min.append(1 / dim + sol["primal objective"] * dim)
            sol = solvers.lp(-c, G, h)
            if not sol["primal objective"]:
                dist_max.append(1)
            else:
                dist_max.append(1 / dim - sol["primal objective"] * dim)

        conf_levels = []
        for delta in deltas:
            conf_levels.append(count_confidence(delta, frequencies, self.tmg.n_measurements))
        self.cl_to_dist_max = interp1d(conf_levels, dist_max)
        self.cl_to_dist_min = interp1d(conf_levels, dist_min)


class PolytopeProcessInterval(ConfidenceInterval):
    def __init__(self, tmg, n_points=1000, target_channel=None):
        """Construct a confidence interval based on linear optimization in a polytope as in work 2109.04734 of
        Kiktenko et al.

        Parameters
        ----------
        tmg : ProcessTomograph
            Object with tomography flat_results
        n_points : int
            Number of distances to get.
        target_channel : qp.Qobj
            If specified, calculates fidelity w.r.t. the Choi matrix of this process
        """
        kwargs = _pop_hidden_keys(locals())
        super().__init__(tmg, **kwargs)

    def __call__(self, conf_levels=None):
        if conf_levels is None:
            conf_levels = np.linspace(1e-3, 1 - 1e-3, 1000)
        if not hasattr(self, "cl_to_dist_max"):
            self.setup()
        return (self.cl_to_dist_min(conf_levels), self.cl_to_dist_max(conf_levels)), conf_levels

    def setup(self):
        channel = self.tmg.channel
        dim_in = dim_out = 2**channel.n_qubits
        dim = dim_in * dim_out
        bloch_indices = [i for i in range(dim**2) if i % (dim_out**2) != 0]

        if self.target_channel is None:
            self.target_channel = channel

        povm_matrix = self.tmg.tomographs[0].povm_matrix
        n_measurements = self.tmg.tomographs[0].n_measurements

        frequencies = np.asarray(
            [np.clip(tmg.results / tmg.n_measurements[:, None], self.EPS, 1 - self.EPS) for tmg in self.tmg.tomographs]
        )

        meas_matrix = (
            np.reshape(
                povm_matrix * n_measurements[:, None, None] / np.sum(n_measurements), (-1, povm_matrix.shape[-1])
            )
            * povm_matrix.shape[0]
        )
        states_matrix = np.asarray([rho.T.bloch for rho in self.tmg.input_basis.elements])
        channel_matrix = np.einsum("i a, j b -> i j a b", states_matrix, meas_matrix[:, 1:]) * dim
        channel_matrix = rearrange(channel_matrix, "i j a b -> (i j) (a b)")
        A = np.ascontiguousarray(channel_matrix)

        max_delta = count_delta(1 - 1e-7, frequencies, n_measurements)
        min_delta = count_delta(0, frequencies, n_measurements)
        deltas = np.linspace(min_delta, max_delta, self.n_points)

        dist_max = []
        dist_min = []
        for delta in deltas:
            b = (
                np.hstack(np.concatenate(frequencies, axis=0))
                + delta
                - repeat(meas_matrix[:, 0], "a -> (b a)", b=len(states_matrix))
            )
            c = matrix(self.target_channel.choi.bloch[bloch_indices])
            G, h = matrix(A), matrix(b)
            sol = solvers.lp(c, G, h)
            if not sol["primal objective"]:
                dist_min.append(1)
            else:
                dist_min.append(1 / dim + sol["primal objective"])
            sol = solvers.lp(-c, G, h)
            if not sol["primal objective"]:
                dist_max.append(1)
            else:
                dist_max.append(1 / dim - sol["primal objective"])

        conf_levels = []
        for delta in deltas:
            conf_levels.append(count_confidence(delta, frequencies, self.tmg.tomographs[0].n_measurements))
        self.cl_to_dist_max = interp1d(conf_levels, dist_max)
        self.cl_to_dist_min = interp1d(conf_levels, dist_min)


# noinspection PyProtectedMember,PyProtectedMember
class HolderInterval(ConfidenceInterval):
    def __init__(
        self,
        tmg,
        n_points=1000,
        kind="wang",
        max_confidence=0.999,
        method="lin",
        method_boot="lin",
        physical=True,
        init="lin",
        tol=1e-3,
        max_iter=100,
        step=0.01,
        burn_steps=1000,
        thinning=1,
    ):
        """Conducts `n_points` experiments, constructs confidence intervals for each,
        computes confidence level that corresponds to the distance between
        the target state and the point estimate and returns a sorted list of these levels.

        Parameters
        ----------
        tmg : ProcessTomograph
            Object with tomography flat_results
        n_points : int
            Number of distances to get.
        kind : str
            Method of constructing the interval.

            Possible values:
                'moment' -- theoretical interval based on moments
                'boot' -- bootstrapping from the point estimate
                'mhmc' -- Metropolis-Hastings Monte Carlo
                'sugiyama' -- 1306.4191 interval
                'wang' -- 1808.09988 interval
        max_confidence : float
            Maximum confidence level for 'moment', 'wang' and 'sugiyama' methods.
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
        """

        kwargs = _pop_hidden_keys(locals())
        super().__init__(tmg, **kwargs)

    def __call__(self, conf_levels=None):
        if conf_levels is None:
            conf_levels = np.linspace(1e-3, 1 - 1e-3, 1000)
        if not hasattr(self, "intervals"):
            self.setup()
        state_results = [interval(conf_levels) for interval in self.intervals]
        state_deltas = np.asarray([state_result[0] for state_result in state_results])
        conf_levels = state_results[0][1] ** self.tmg.input_basis.dim

        coef = np.abs(
            np.einsum(
                "ij,ik->jk",
                self.tmg._decomposed_single_entries,
                self.tmg._decomposed_single_entries.conj(),
            )
        )
        state_deltas_composition = np.einsum("ik,jk->ijk", state_deltas, state_deltas)
        dist = np.sqrt(np.einsum("ijk,ij->k", state_deltas_composition, coef))
        return dist, conf_levels

    def setup(self):
        if self.mode == Mode.STATE:
            raise NotImplementedError("Holder interval works only for process tomography")
        if self.kind == "moment":
            self.intervals = [MomentInterval(tmg, self.n_points, self.max_confidence) for tmg in self.tmg.tomographs]
        elif self.kind == "mhmc":
            self.intervals = [
                MHMCStateInterval(tmg, self.n_points, self.step, self.burn_steps, self.thinning)
                for tmg in self.tmg.tomographs
            ]
        elif self.kind == "bootstrap":
            self.intervals = [
                BootstrapStateInterval(
                    tmg,
                    self.n_points,
                    self.method,
                    physical=self.physical,
                    init=self.init,
                    tol=self.tol,
                    max_iter=self.max_iter,
                )
                for tmg in self.tmg.tomographs
            ]
        elif self.kind == "sugiyama":
            self.intervals = [SugiyamaInterval(tmg, self.n_points, self.max_confidence) for tmg in self.tmg.tomographs]
        else:
            raise ValueError("Incorrect value for argument `kind`.")

        for interval in self.intervals:
            interval.setup()


class BootstrapStateInterval(ConfidenceInterval):
    def __init__(
        self,
        tmg,
        n_points=1000,
        method="lin",
        physical=True,
        init="lin",
        tol=1e-3,
        max_iter=100,
        state=None,
    ):
        """Perform multiple tomography simulation on the preferred state with the same
        measurements number
        and POVM matrix, as in the preceding experiment. Count the distances to the
        bootstrapped
        states.

        Parameters
        ----------
        tmg : StateTomograph
            Object with tomography flat_results
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
        kwargs = _pop_hidden_keys(locals())
        super().__init__(tmg, **kwargs)

    def setup(self):
        if self.mode == Mode.CHANNEL:
            raise NotImplementedError("This interval works only for state tomography")
        if self.state is None:
            if hasattr(self.tmg, "reconstructed_state"):
                self.state = self.tmg.reconstructed_state
            else:
                self.state = self.tmg.point_estimate(
                    method=self.method,
                    physical=self.physical,
                    init=self.init,
                    tol=self.tol,
                    max_iter=self.max_iter,
                )

        dist = np.empty(self.n_points)
        boot_tmg = self.tmg.__class__(self.state, self.tmg.dst)
        for i in range(self.n_points):
            boot_tmg.experiment(self.tmg.n_measurements, self.tmg.povm_matrix)
            rho = boot_tmg.point_estimate(
                method=self.method,
                physical=self.physical,
                init=self.init,
                tol=self.tol,
                max_iter=self.max_iter,
            )
            dist[i] = self.tmg.dst(rho, self.state)
        dist.sort()
        conf_levels = np.linspace(0, 1, len(dist))
        self.cl_to_dist = interp1d(conf_levels, dist)


class BootstrapProcessInterval(ConfidenceInterval):
    def __init__(
        self,
        tmg,
        n_points=1000,
        method="lifp",
        cptp=True,
        tol=1e-10,
        channel=None,
        states_est_method="lin",
        states_physical=True,
        states_init="lin",
    ):
        """Perform multiple tomography simulation on the preferred channel with the same
        measurements number
        and POVM matrix, as in the preceding experiment. Count the distances to the
        bootstrapped
        Choi matrices.

        Parameters
        ----------
        tmg : ProcessTomograph
            Object with tomography flat_results
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
        kwargs = _pop_hidden_keys(locals())
        super().__init__(tmg, **kwargs)

    def setup(self):
        if self.mode == Mode.STATE:
            raise NotImplementedError("This interval works only for process tomography")
        if self.channel is None:
            if hasattr(self.tmg, "reconstructed_channel"):
                self.channel = self.tmg.reconstructed_channel
            else:
                self.channel = self.tmg.point_estimate(
                    method=self.method,
                    states_physical=self.states_physical,
                    states_init=self.states_init,
                    cptp=self.cptp,
                )

        dist = np.empty(self.n_points)
        boot_tmg = self.tmg.__class__(self.channel, self.tmg.input_states, self.tmg.dst)
        for i in range(self.n_points):
            boot_tmg.experiment(self.tmg.tomographs[0].n_measurements, povm=self.tmg.tomographs[0].povm_matrix)
            estim_channel = boot_tmg.point_estimate(
                method=self.method,
                states_physical=self.states_physical,
                states_init=self.states_init,
                cptp=self.cptp,
            )
            dist[i] = self.tmg.dst(estim_channel.choi, self.channel.choi)
        dist.sort()
        conf_levels = np.linspace(0, 1, len(dist))
        self.cl_to_dist = interp1d(conf_levels, dist)


# noinspection PyProtectedMember
class MHMCStateInterval(ConfidenceInterval):
    def __init__(
        self,
        tmg,
        n_points=1000,
        step=0.01,
        burn_steps=1000,
        thinning=1,
        warm_start=False,
        use_new_estimate=False,
        state=None,
        verbose=False,
    ):
        """Use Metropolis-Hastings Monte Carlo algorithm to obtain samples from likelihood
        distribution.
        Count the distances between these samples and point estimate.

        Parameters
        ----------
        tmg : StateTomograph
            Object with tomography flat_results
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
        kwargs = _pop_hidden_keys(locals())
        super().__init__(tmg, **kwargs)

    # noinspection PyTypeChecker
    def setup(self):
        if self.mode == Mode.CHANNEL:
            raise NotImplementedError("This interval works only for state tomography")
        if not self.use_new_estimate:
            self.state = self.tmg.reconstructed_state
        elif self.state is None:
            self.state = self.tmg.point_estimate(method="mle", physical=True)

        dim = 4**self.tmg.state.n_qubits
        if not (self.warm_start and hasattr(self, "chain")):
            x_init = _matrix_to_real_tril_vec(self.state.matrix)
            self.chain = MHMC(
                lambda x: -self.tmg._nll(x),
                step=self.step,
                burn_steps=self.burn_steps,
                dim=dim,
                update_rule=normalized_update,
                symmetric=True,
                x_init=x_init,
            )
        samples, acceptance_rate = self.chain.sample(self.n_points, self.thinning, verbose=self.verbose)
        dist = np.asarray([self.tmg.dst(_real_tril_vec_to_matrix(tril_vec), self.state.matrix) for tril_vec in samples])
        dist.sort()
        conf_levels = np.linspace(0, 1, len(dist))
        self.cl_to_dist = interp1d(conf_levels, dist)


# noinspection PyProtectedMember,PyProtectedMember
class MHMCProcessInterval(ConfidenceInterval):
    def __init__(
        self,
        tmg,
        n_points=1000,
        step=0.01,
        burn_steps=1000,
        thinning=1,
        warm_start=False,
        method="lifp",
        states_est_method="lin",
        states_physical=True,
        states_init="lin",
        use_new_estimate=False,
        channel=None,
        verbose=False,
        return_samples=False,
    ):
        """Use Metropolis-Hastings Monte Carlo algorithm to obtain samples from likelihood
        distribution.
        Count the distances between these samples and point estimate.

        Parameters
        ----------
        tmg : ProcessTomograph
            Object with tomography flat_results
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
        kwargs = _pop_hidden_keys(locals())
        super().__init__(tmg, **kwargs)

    # noinspection PyTypeChecker
    def setup(self):
        if self.mode == Mode.STATE:
            raise NotImplementedError("This interval works only for process tomography")
        if not self.use_new_estimate:
            self.channel = self.tmg.reconstructed_channel
        elif self.channel is None:
            self.channel = self.tmg.point_estimate(
                self.method,
                states_est_method=self.states_est_method,
                states_physical=self.states_physical,
                states_init=self.states_init,
            )

        dim = 16**self.tmg.channel.n_qubits
        if not (self.warm_start and hasattr(self, "chain")):
            x_init = _mat2vec(self.channel.choi.matrix)
            self.chain = MHMC(
                lambda x: -self.tmg._nll(x),
                step=self.step,
                burn_steps=self.burn_steps,
                dim=dim,
                update_rule=self.tmg._cptp_update_rule,
                symmetric=True,
                x_init=x_init,
            )
        samples, acceptance_rate = self.chain.sample(self.n_points, self.thinning, verbose=self.verbose)
        dist = np.asarray([self.tmg.dst(_vec2mat(choi_vec), self.channel.choi.matrix) for choi_vec in samples])
        dist.sort()
        conf_levels = np.linspace(0, 1, len(dist))
        if self.return_samples:
            matrices = [_vec2mat(choi_vec) for choi_vec in samples]
            return dist, conf_levels, acceptance_rate, matrices
        self.cl_to_dist = interp1d(conf_levels, dist)


class Mode(Enum):
    STATE = auto()
    CHANNEL = auto()


def _pop_hidden_keys(kwargs):
    keys_to_pop = ["self", "tmg"]
    for key in kwargs.keys():
        if key.startswith("__"):
            keys_to_pop.append(key)
    for key in keys_to_pop:
        kwargs.pop(key)
    return kwargs
