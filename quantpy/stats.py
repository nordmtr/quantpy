import numpy as np
from einops import rearrange


def l2_mean(freq, n_trials, weights=None):
    """Return the mean of the squared l2-norm of a vector (f-p), where `f` is an MLE estimate
    of the `p` parameter of the multinomial distribution with `n_trials`."""
    if weights is None:
        weights = make_identity_weights(freq)
    return l2_first_moment(freq, n_trials, weights)


def l2_variance(freq, n_trials, weights=None):
    """Return the variance of the squared l2-norm of a vector (f-p), where `f` is an MLE estimate
    of the `p` parameter of the multinomial distribution with `n_trials`."""
    if weights is None:
        weights = make_identity_weights(freq)
    return l2_second_moment(freq, n_trials, weights) - l2_first_moment(freq, n_trials, weights) ** 2


def l2_first_moment(freq, n_trials, weights):
    """Return the first raw moment of the squared l2-norm of a vector (f-p), where `f` is an MLE
    estimate
    of the `p` parameter of the multinomial distribution with `n_trials`."""
    return (np.einsum("aiai,ai->", weights, freq) - np.einsum("aiaj,ai,aj->", weights, freq, freq)) / n_trials


def l2_second_moment(freq, n_trials, weights):
    """Return the second raw moment of the squared l2-norm of a vector (f-p), where `f` is an MLE
    estimate
    of the `p` parameter of the multinomial distribution with `n_trials`."""
    return (
        # Non-zero exp value only if from one povm, or split 2/2 between povms.
        # Variance for the former is exactly equal to the one in the latter substituting second povm with first one.
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
    ) / n_trials**2


def make_identity_weights(freq):
    weights = np.tensordot(np.eye(freq.shape[0]), np.eye(freq.shape[1]), axes=0)
    weights = rearrange(weights, "a b c d -> a c b d")
    return weights
