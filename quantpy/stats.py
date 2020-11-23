import numpy as np


def l2_first_moment(frequencies, n_trials):
    """Return the first raw moment of the squared l2-norm of a vector (f-p), where `f` is an MLE estimate
    of the `p` parameter of the multinomial distribution with `n_trials`."""
    return np.einsum('ki->k', frequencies - frequencies ** 2) / n_trials


def l2_second_moment(frequencies, n_trials):
    """Return the second raw moment of the squared l2-norm of a vector (f-p), where `f` is an MLE estimate
    of the `p` parameter of the multinomial distribution with `n_trials`."""
    return (
        3 * np.einsum('ki,kj->k', frequencies ** 2, frequencies ** 2)
        + np.einsum('ki,kj->k', frequencies, frequencies)
        - np.einsum('ki,kj->k', frequencies ** 2, frequencies)
        - np.einsum('ki,kj->k', frequencies, frequencies ** 2)
        - 4 * np.einsum('ki->k', frequencies ** 3)
        + 2 * np.einsum('ki->k', frequencies ** 2)
    ) / n_trials ** 2


def l2_mean(frequencies, n_trials):
    """Return the mean of the squared l2-norm of a vector (f-p), where `f` is an MLE estimate
    of the `p` parameter of the multinomial distribution with `n_trials`."""
    return l2_first_moment(frequencies, n_trials)


def l2_variance(frequencies, n_trials):
    """Return the variance of the squared l2-norm of a vector (f-p), where `f` is an MLE estimate
    of the `p` parameter of the multinomial distribution with `n_trials`."""
    return ((
        3 * np.einsum('ki,kj->k', frequencies ** 2, frequencies ** 2)
        + np.einsum('ki,kj->k', frequencies, frequencies)
        - np.einsum('ki,kj->k', frequencies ** 2, frequencies)
        - np.einsum('ki,kj->k', frequencies, frequencies ** 2)
        - 4 * np.einsum('ki->k', frequencies ** 3)
        + 2 * np.einsum('ki->k', frequencies ** 2)
    ) - np.einsum('ki->k', frequencies - frequencies ** 2) ** 2) / n_trials ** 2
