import numpy as np
import scipy.linalg as la

from .geometry import hs_dst, if_dst, trace_dst, product
from .qobj import Qobj
from .channel import Channel
from .tomography import Tomograph
from .measurements import generate_measurement_matrix
from .basis import Basis


class ProcessTomograph:
    def __init__(self, channel, dst='hs', input_states='sic'):
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
        self.input_states_list = _generate_input_states(input_states)
        if len(self.input_states_list) != 4 ** self.input_states_list[0].n_qubits:
            raise ValueError('Input states do not constitute a basis')
        self._decomposed_single_entries = _decompose_se(self.input_states_list)

    def experiment(self, n_measurements, POVM='proj', method='lin', physical=True, init='lin'):
        output_states = []
        for input_state in self.input_states_list:
            output_state = self.channel.transform(input_state)
            tmg = Tomograph(output_state)
            tmg.experiment(n_measurements, POVM)
            output_states.append(tmg.point_estimate(method, physical, init))

    def point_estimate(self, physical=True):
        pass

    def bootstrap(self, n_boot):
        pass


def _generate_input_states(type='sic', n_qubits=1):
    input_states_list = []
    for input_state_bloch in generate_measurement_matrix(type, n_qubits):
        input_state = Qobj(input_state_bloch)
        input_state /= input_state.trace()
        input_states_list.append(input_state)
    return input_states_list


def _decompose_se(states_list):  # TODO
    basis
