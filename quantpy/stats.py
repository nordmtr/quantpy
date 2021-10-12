import numpy as np


def l2_mean(freq, n_trials, inv_povm_matrix=None):
    """Return the mean of the squared l2-norm of a vector (f-p), where `f` is an MLE estimate
    of the `p` parameter of the multinomial distribution with `n_trials`."""
    if inv_povm_matrix is None:
        weights = np.eye(freq.shape[-1])
    else:
        weights = inv_povm_matrix.T.conj() @ inv_povm_matrix
    # print((
    #     np.einsum("ii,i->", weights, freq)
    #     - np.einsum("ij,i,j->", weights, freq, freq)
    # ) / n_trials)
    return (
        np.einsum("ii,i->", weights, freq)
        - np.einsum("ij,i,j->", weights, freq, freq)
    ) / n_trials


def l2_variance(freq, n_trials, inv_povm_matrix=None):
    """Return the variance of the squared l2-norm of a vector (f-p), where `f` is an MLE estimate
    of the `p` parameter of the multinomial distribution with `n_trials`."""
    if inv_povm_matrix is None:
        weights = np.eye(freq.shape[-1])
    else:
        weights = inv_povm_matrix.T.conj() @ inv_povm_matrix
    return (
        3 * np.einsum("ij,kl,i,j,k,l->", weights, weights, freq, freq, freq, freq)

        - np.einsum("ii,kl,i,k,l->", weights, weights, freq, freq, freq)  # i = j
        - np.einsum("ij,il,i,j,l->", weights, weights, freq, freq, freq)
        - np.einsum("ij,ki,i,j,k->", weights, weights, freq, freq, freq)
        - np.einsum("ij,jl,i,j,l->", weights, weights, freq, freq, freq)
        - np.einsum("ij,kj,i,j,k->", weights, weights, freq, freq, freq)
        - np.einsum("ij,kk,i,j,k->", weights, weights, freq, freq, freq)

        # + np.einsum("ii,il,i,l->", weights, weights, freq, freq)  # i = j = k
        # + np.einsum("ii,ki,i,k->", weights, weights, freq, freq)
        # + np.einsum("ij,ii,i,j->", weights, weights, freq, freq)
        # + np.einsum("ij,jj,i,j->", weights, weights, freq, freq)

        + np.einsum("ii,kk,i,k->", weights, weights, freq, freq)  # i = j & k = l
        + np.einsum("ij,ij,i,j->", weights, weights, freq, freq)
        + np.einsum("ij,ji,i,j->", weights, weights, freq, freq)

        # + np.einsum("ii,ii,i->", weights, weights, freq)  # i = j = k = l
    ) / n_trials ** 2 - l2_mean(freq, n_trials, inv_povm_matrix) ** 2


def l2_first_moment(frequencies, n_trials):
    """Return the first raw moment of the squared l2-norm of a vector (f-p), where `f` is an MLE
    estimate
    of the `p` parameter of the multinomial distribution with `n_trials`."""

    return np.einsum("ki->k", frequencies - frequencies ** 2) / n_trials


def l2_second_moment(frequencies, n_trials):
    """Return the second raw moment of the squared l2-norm of a vector (f-p), where `f` is an MLE
    estimate
    of the `p` parameter of the multinomial distribution with `n_trials`."""
    return (
        3 * np.einsum("ki,kj->k", frequencies ** 2, frequencies ** 2)
        + np.einsum("ki,kj->k", frequencies, frequencies)
        - np.einsum("ki,kj->k", frequencies ** 2, frequencies)
        - np.einsum("ki,kj->k", frequencies, frequencies ** 2)
        - 4 * np.einsum("ki->k", frequencies ** 3)
        + 2 * np.einsum("ki->k", frequencies ** 2)
    ) / n_trials ** 2


# def l2_mean(frequencies, n_trials):
#     """Return the mean of the squared l2-norm of a vector (f-p), where `f` is an MLE estimate
#     of the `p` parameter of the multinomial distribution with `n_trials`."""
#     return l2_first_moment(frequencies, n_trials, weights_matrix)
#
#
# def l2_variance(frequencies, n_trials):
#     """Return the variance of the squared l2-norm of a vector (f-p), where `f` is an MLE estimate
#     of the `p` parameter of the multinomial distribution with `n_trials`."""
#     # return ((
#     #                 3 * np.einsum('ki,kj->k', frequencies ** 2, frequencies ** 2)
#     #                 + np.einsum('ki,kj->k', frequencies, frequencies)
#     #                 - np.einsum('ki,kj->k', frequencies ** 2, frequencies)
#     #                 - np.einsum('ki,kj->k', frequencies, frequencies ** 2)
#     #                 - 4 * np.einsum('ki->k', frequencies ** 3)
#     #                 + 2 * np.einsum('ki->k', frequencies ** 2)
#     #         ) - np.einsum('ki->k', frequencies - frequencies ** 2) ** 2) / n_trials ** 2
#     return l2_second_moment(frequencies, n_trials) - l2_first_moment(frequencies, n_trials) ** 2
