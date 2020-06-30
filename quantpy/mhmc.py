import numpy as np
# import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class MHMC:
    """Metropolis-Hastings algorithm for generating samples from an unnormalized distribution.

    Parameters
    ----------
    target_logpdf : function
        Unnormalized target log-probability density function.
    jump_distr : scipy.stats.rv_continuous object or None
        Jumping distribution. Should have `pdf` and `rvs` methods.
        If None, standard normal distribution is used.
    step : float
        Multiplier used in each step.
    burn_steps : int
        Steps for burning in.
    dim : int
        Dimension of the random variable.
    update_rule : function or None
        Takes current `x_t`, `delta` and `step` as arguments and returns the proposed `x_prime`.
        If None, x_prime = x_t + step * delta
    symmetric : bool
        If True, assumes jump_distr to be symmetric.
    """

    def __init__(self, target_logpdf, jump_distr=None, step=0.01, burn_steps=100,
                 dim=1, update_rule=None, symmetric=False):
        self.target_logpdf = target_logpdf
        self.jump_distr = jump_distr if jump_distr is not None else multivariate_normal(mean=np.zeros(dim))
        self.step = step
        self.burn_steps = burn_steps
        self.dim = dim
        self.update_rule = update_rule if update_rule is not None else basic_update
        self.symmetric = symmetric

        self.x_t = np.random.rand(dim)
        self.burned = False

    def sample(self, n_samples, thinning=1):
        if not self.burned:
            self._burn_in()

        total_n_samples = n_samples * thinning
        deltas = self.jump_distr.rvs(size=total_n_samples)
        random_values = np.random.rand(total_n_samples)
        samples = np.zeros((n_samples, self.dim))
        accepted_samples = 0

        for i in range(total_n_samples):
            is_accepted = self._step(deltas[i], random_values[i])
            accepted_samples += is_accepted
            if i % thinning == 0:
                samples[i // thinning] = self.x_t

        acceptance_rate = accepted_samples / total_n_samples

        return samples, acceptance_rate

    def _burn_in(self):
        deltas = self.jump_distr.rvs(size=self.burn_steps)
        random_values = np.random.rand(self.burn_steps)

        for i in range(self.burn_steps):
            self._step(deltas[i], random_values[i])

        self.burned = True

    def _step(self, delta, random_value):
        x_prime = self.update_rule(self.x_t, delta, self.step)
        alpha = np.exp(self.target_logpdf(x_prime) - self.target_logpdf(self.x_t))
        if not self.symmetric:
            alpha *= self.jump_distr.pdf(-delta) / self.jump_distr.pdf(delta)
        if random_value <= alpha:
            self.x_t = x_prime
            is_accepted = True
        else:
            is_accepted = False

        return is_accepted


def basic_update(x_t, delta, step):
    return x_t + step * delta


def normalized_update(x_t, delta, step):
    unnorm_x_prime = x_t + step * delta
    return unnorm_x_prime / np.linalg.norm(unnorm_x_prime)
