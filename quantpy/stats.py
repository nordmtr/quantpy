import numpy as np
from einops import rearrange


def l2_mean(freq, n_trials, weights=None):
    """Return the mean of the squared l2-norm of a vector (f-p), where `f` is an MLE estimate
    of the `p` parameter of the multinomial distribution with `n_trials`."""
    if weights is None:
        weights = np.tensordot(np.eye(freq.shape[0]), np.eye(freq.shape[1]), axes=0)
        weights = rearrange(weights, "a b c d -> a c b d")
    return (
        np.einsum("aiai,ai->", weights, freq) - np.einsum("aiaj,ai,aj->", weights, freq, freq)
    ) / n_trials


def l2_variance(freq, n_trials, weights=None):
    """Return the variance of the squared l2-norm of a vector (f-p), where `f` is an MLE estimate
    of the `p` parameter of the multinomial distribution with `n_trials`."""
    if weights is None:
        weights = np.tensordot(np.eye(freq.shape[0]), np.eye(freq.shape[1]), axes=0)
        weights = rearrange(weights, "a b c d -> a c b d")
    return (
        # all freqs from one povm
        # 3 * np.einsum("aiaj,akal,ai,aj,ak,al->", weights, weights, freq, freq, freq, freq)
        #
        # - np.einsum("aiai,akal,ai,ak,al->", weights, weights, freq, freq, freq)  # i = j
        # - np.einsum("aiaj,aial,ai,aj,al->", weights, weights, freq, freq, freq)
        # - np.einsum("aiaj,akai,ai,aj,ak->", weights, weights, freq, freq, freq)
        # - np.einsum("aiaj,ajal,ai,aj,al->", weights, weights, freq, freq, freq)
        # - np.einsum("aiaj,akaj,ai,aj,ak->", weights, weights, freq, freq, freq)
        # - np.einsum("aiaj,akak,ai,aj,ak->", weights, weights, freq, freq, freq)
        #
        # + np.einsum("aiai,akak,ai,ak->", weights, weights, freq, freq)  # i = j & k = l
        # + np.einsum("aiaj,aiaj,ai,aj->", weights, weights, freq, freq)
        # + np.einsum("aiaj,ajai,ai,aj->", weights, weights, freq, freq)
        # freqs split 2/2 between povms
        np.einsum("aiaj,bkbl,ai,aj,bk,bl->", weights, weights, freq, freq, freq, freq)
        - np.einsum("aiaj,bkbk,ai,aj,bk->", weights, weights, freq, freq, freq)
        - np.einsum("aiai,bkbl,ai,bk,bl->", weights, weights, freq, freq, freq)
        + np.einsum("aiai,bkbk,ai,bk->", weights, weights, freq, freq)
        + np.einsum("aibj,bkal,ai,bj,bk,al->", weights, weights, freq, freq, freq, freq)
        - np.einsum("aibj,bjal,ai,bj,al->", weights, weights, freq, freq, freq)
        - np.einsum("aibj,bkai,ai,bj,bk->", weights, weights, freq, freq, freq)
        + np.einsum("aibj,bjai,ai,bj->", weights, weights, freq, freq)
        + np.einsum("aibj,akbl,ai,bj,ak,bl->", weights, weights, freq, freq, freq, freq)
        - np.einsum("aibj,akbj,ai,bj,ak->", weights, weights, freq, freq, freq)
        - np.einsum("aibj,aibl,ai,bj,bl->", weights, weights, freq, freq, freq)
        + np.einsum("aibj,aibj,ai,bj->", weights, weights, freq, freq)
    ) / n_trials ** 2 - l2_mean(freq, n_trials, weights) ** 2


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
