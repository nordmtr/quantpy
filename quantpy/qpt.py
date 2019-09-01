import numpy as np

from .geometry import hs_dst, if_dst, trace_dst
from .routines import generate_single_entries, kron
from .qobj import Qobj
from .channel import Channel
from .tomography import Tomograph
from .measurements import generate_measurement_matrix
from .basis import Basis


class ProcessTomograph:
    def __init__(self, channel, dst='hs', input_states='proj4'):
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
        self.input_basis = Basis(_generate_input_states(input_states))
        if self.input_basis.dim != 4 ** self.channel.n_qubits:
            raise ValueError('Input states do not constitute a basis')
        self._decomposed_single_entries = np.array([
            self.input_basis.decompose(Qobj(single_entry))
            for single_entry in generate_single_entries(2 ** self.channel.n_qubits)
        ])

    def experiment(self, n_measurements, POVM='proj', warm_start=False):
        if not warm_start:
            self.tomographs = []
            for input_state in self.input_basis.elements:
                output_state_true = self.channel.transform(input_state)
                tmg = Tomograph(output_state_true)
                self.tomographs.append(tmg)
        for tmg in self.tomographs:
            tmg.experiment(n_measurements, POVM)

    def point_estimate(self, method='lin', physical=True, init='lin'):
        output_states = [
            tmg.point_estimate(method, physical, init) for tmg in self.tomographs
        ]
        output_basis = Basis(output_states)
        choi_matrix = Qobj(np.zeros((output_basis.dim, output_basis.dim)))
        for decomposed_single_entry in self._decomposed_single_entries:
            single_entry = self.input_basis.compose(decomposed_single_entry)
            transformed_single_entry = output_basis.compose(decomposed_single_entry)
            choi_matrix += kron(single_entry, transformed_single_entry)
        return Channel(choi_matrix)

    def bootstrap(self, n_boot):
        pass


def _generate_input_states(type='proj4', n_qubits=1):
    """Generate input states to use in quantum process tomography"""
    input_states_list = []
    for input_state_bloch in generate_measurement_matrix(type, n_qubits):
        input_state = Qobj(input_state_bloch)
        input_state /= input_state.trace()
        input_states_list.append(input_state)
    return input_states_list
