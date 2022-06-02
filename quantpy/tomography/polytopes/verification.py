import numpy as np
from einops import rearrange, repeat
from tqdm.auto import tqdm

from quantpy import ProcessTomograph, StateTomograph
from quantpy.tomography.polytopes.utils import count_delta


def test_qst(state, conf_levels, n_measurements=1000, n_trials=1000):
    results = np.zeros(len(conf_levels))

    dim = 2**state.n_qubits
    tmg = StateTomograph(state)
    tmg.experiment(n_measurements)
    EPS = 1e-15

    povm_matrix = (
        np.reshape(
            tmg.povm_matrix * tmg.n_measurements[:, None, None] / np.sum(tmg.n_measurements),
            (-1, tmg.povm_matrix.shape[-1]),
        )
        * tmg.povm_matrix.shape[0]
    )
    A = np.ascontiguousarray(povm_matrix[:, 1:]) * dim
    polytope_prod = A @ state.bloch[1:]

    for _ in tqdm(range(n_trials)):
        tmg = StateTomograph(state)
        tmg.experiment(n_measurements)
        frequencies = np.clip(tmg.results / tmg.n_measurements[:, None], EPS, 1 - EPS)
        for j, cl in enumerate(conf_levels):
            delta = count_delta(cl, frequencies, tmg.n_measurements)
            b = np.clip(np.hstack(frequencies) + delta, EPS, 1 - EPS) - povm_matrix[:, 0]
            if np.min(b - polytope_prod) > -EPS:
                results[j] += 1
    results /= n_trials
    return results


def test_qpt(channel, conf_levels, n_measurements=1000, n_trials=1000, input_states="sic"):
    results = np.zeros(len(conf_levels))

    dim = 4**channel.n_qubits
    bloch_indices = [i for i in range(dim**2) if i % dim != 0]
    tmg = ProcessTomograph(channel, input_states=input_states)
    tmg.experiment(n_measurements)
    EPS = 1e-15

    povm_matrix = tmg.tomographs[0].povm_matrix
    n_measurements = tmg.tomographs[0].n_measurements

    meas_matrix = (
        np.reshape(povm_matrix * n_measurements[:, None, None] / np.sum(n_measurements), (-1, povm_matrix.shape[-1]))
        * povm_matrix.shape[0]
    )
    states_matrix = np.asarray([rho.T.bloch for rho in tmg.input_basis.elements])
    channel_matrix = np.einsum("i a, j b -> i j a b", states_matrix, meas_matrix[:, 1:]) * dim
    channel_matrix = rearrange(channel_matrix, "i j a b -> (i j) (a b)")
    A = np.ascontiguousarray(channel_matrix)
    polytope_prod = A @ channel.choi.bloch[bloch_indices]

    for _ in tqdm(range(n_trials)):
        tmg = ProcessTomograph(channel, input_states=input_states)
        tmg.experiment(n_measurements)
        frequencies = np.asarray(
            [np.clip(ptmg.results / ptmg.n_measurements[:, None], EPS, 1 - EPS) for ptmg in tmg.tomographs]
        )
        for j, cl in enumerate(conf_levels):
            delta = count_delta(cl, frequencies, tmg.tomographs[0].n_measurements)
            b = (
                np.hstack(np.concatenate(frequencies, axis=0))
                + delta
                - repeat(meas_matrix[:, 0], "a -> (b a)", b=len(states_matrix))
            )
            if np.min(b - polytope_prod) > -EPS:
                results[j] += 1
    results /= n_trials
    return results
