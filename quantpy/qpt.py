import numpy as np

from scipy.optimize import minimize

from .geometry import hs_dst, if_dst, trace_dst
from .routines import generate_single_entries, kron, _real_tril_vec_to_matrix, _matrix_to_real_tril_vec
from .qobj import Qobj, fully_mixed
from .channel import Channel, depolarizing
from .tomography import Tomograph
from .measurements import generate_measurement_matrix
from .basis import Basis


def _tp_constraint(tril_vec):
    choi_matrix = Qobj(_real_tril_vec_to_matrix(tril_vec))
    rho_in = choi_matrix.ptrace(list(range(choi_matrix.n_qubits // 2)))
    return hs_dst(rho_in.matrix, np.eye(2 ** rho_in.n_qubits))


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
        List of Tomograph objects corresponding to each input state

    Methods
    -------
    bootstrap()
        Perform multiple tomography simulation
    experiment()
        Simulate a real quantum process tomography
    point_estimate()
        Reconstruct a channel from the data obtained in the experiment
    """
    def __init__(self, channel, input_states='proj4', dst='hs', input_impurity=0.05):
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
        self.input_basis = Basis(_generate_input_states(input_states, input_impurity, channel.n_qubits))
        if self.input_basis.dim != 4 ** channel.n_qubits:
            raise ValueError('Input states do not constitute a basis')
        self._decomposed_single_entries = np.array([
            self.input_basis.decompose(Qobj(single_entry))
            for single_entry in generate_single_entries(2 ** channel.n_qubits)
        ])

    def experiment(self, n_measurements, POVM='proj', warm_start=False):
        """Simulate a real quantum process tomography by performing
        quantum state tomography on each of transformed input states.

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
        if not warm_start:
            self.tomographs = []
            for input_state in self.input_basis.elements:
                output_state_true = self.channel.transform(input_state)
                tmg = Tomograph(output_state_true)
                self.tomographs.append(tmg)
        for tmg in self.tomographs:
            tmg.experiment(n_measurements, POVM)

    def point_estimate(self, method='lin', physical=True, init='lin', cptp=True):
        """Reconstruct a Choi matrix from the data obtained in the experiment

        Parameters
        ----------
        method : str, default='lin'
            Method of reconstructing of every output state

            Possible values:
                'lin' -- linear inversion
                'mle' -- maximum likelihood estimation with Cholesky parametrization, unconstrained optimization
                'mle-constr' -- same as 'mle', but optimization is constrained
                'mle-bloch' -- maximum likelihood estimation with Bloch parametrization,
                               constrained optimization (works only for 1-qubit systems)

        physical : bool, default=True (optional)
            For methods 'lin' and 'mle' reconstructed matrix of output state may not lie in the physical domain.
            If True, set negative eigenvalues to zeros and divide the matrix by its trace.

        init : str, default='lin' (optional)
            Methods using maximum likelihood estimation require the starting point for gradient descent.

            Possible values:
                'lin' -- uses linear inversion point estimate as initial guess
                'mixed' -- uses fully mixed state as initial guess

        cptp : bool, default=True
            If True, return a projection onto CPTP space.

        Returns
        -------
        reconstructed_channel : Channel
        """
        output_states = [
            tmg.point_estimate(method, physical, init) for tmg in self.tomographs
        ]
        output_basis = Basis(output_states)
        choi_matrix = Qobj(np.zeros((output_basis.dim, output_basis.dim)))
        for decomposed_single_entry in self._decomposed_single_entries:
            single_entry = self.input_basis.compose(decomposed_single_entry)
            transformed_single_entry = output_basis.compose(decomposed_single_entry)
            choi_matrix += kron(single_entry, transformed_single_entry)
        self.reconstructed_channel = Channel(choi_matrix)
        if cptp and not self.reconstructed_channel.is_cptp():
            x0 = fully_mixed(choi_matrix.n_qubits).matrix
            x0 = _matrix_to_real_tril_vec(x0)
            constraints = [
                {'type': 'eq', 'fun': _tp_constraint},
            ]
            opt_res = minimize(
                lambda x: hs_dst(choi_matrix, _real_tril_vec_to_matrix(x)),
                x0, constraints=constraints, method='SLSQP'
            )
            choi_matrix = _real_tril_vec_to_matrix(opt_res.x)
        self.reconstructed_channel = Channel(choi_matrix)
        return self.reconstructed_channel

    def bootstrap(self, n_boot, est_method='lin', physical=True, init='lin',
                  use_new_estimate=False, channel=None, kind='estim', cptp=True):
        """Perform multiple tomography simulation on the preferred channel with the same measurements number
        and POVM matrix, as in the preceding experiment. Count the distances to the bootstrapped Choi matrices.

        Parameters
        ----------
        n_boot : int
            Number of experiments to perform
        est_method : str, default='lin'
            Method of reconstructing the density matrix for each output state
            See :ref:`point_estimate` for detailed documentation
        physical : bool, default=True (optional)
            See :ref:`point_estimate` for detailed documentation
        init : str, default='lin' (optional)
            See :ref:`point_estimate` for detailed documentation
        use_new_estimate : bool, default=False
            If False, uses the latest reconstructed channel as a channel to perform new tomographies on.
            If True and `channel` is None, reconstruct a density matrix from the data obtained in previous experiment
            ans use it to perform new tomographies on.
            If True and `channel` is not None, use `channel` as a channel to perform new tomographies on.
        channel : Qobj or None, default=None
            If not None and `use_new_estimate` is True, use it as a channel to perform new tomographies on
        kind : str, default='est'
            Type of confidence interval to build.
            Possible values:
                'estim' -- CI for the point estimate
                'target' -- CI for the target channel built with bootstrap from point estimate only
                'triangle' -- CI for the target channel built with bootstrap from point estimate only
                              + triangle inequality
        cptp : bool, default=True
            If True, all bootstrap samples are projected onto CPTP space
        """
        if not use_new_estimate:
            channel = self.reconstructed_channel
        elif channel is None:
            channel = self.point_estimate(method=est_method, physical=physical, init=init, cptp=cptp)

        dist = [0]
        boot_tmg = self.__class__(channel, self.input_states, self.dst)
        for _ in range(n_boot):
            boot_tmg.experiment(self.tomographs[0].n_measurements, POVM=self.tomographs[0].POVM_matrix)
            estim_channel = boot_tmg.point_estimate(method=est_method, physical=physical, init=init, cptp=cptp)
            if kind == 'estim':
                dist.append(self.dst(estim_channel.choi, channel.choi))
            elif kind == 'target':
                dist.append(self.dst(estim_channel.choi, self.channel.choi))
            elif kind == 'triangle':
                dist.append(self.dst(estim_channel.choi, channel.choi) + self.dst(channel.choi, self.channel.choi))
            else:
                raise ValueError('Invalid value for argument `kind`')
        dist.sort()
        return dist


def _generate_input_states(type, input_impurity, n_qubits):
    """Generate input states to use in quantum process tomography"""
    input_states_list = []
    for input_state_bloch in generate_measurement_matrix(type, n_qubits):
        input_state = Qobj(input_state_bloch)
        input_state /= input_state.trace()
        input_state = depolarizing(input_impurity, n_qubits).transform(input_state)
        input_states_list.append(input_state)
    return input_states_list
