from scipy.stats import uniform, norm
import numpy as np


def draw_uniform(a: float = 0.0, b: float = 1.0, size: int = 1) -> np.ndarray:
    """
    draw random samples from uniform distribution

    :param a: minimum value
    :param b: maximum value
    :param size: number of random samples tb drawn
    :return: sampled values
    """
    scale = b - a
    return uniform.rvs(loc=a, scale=scale, size=size)


def draw_normal(mu: float = 0.0, sigma: float = 1.0, size: int = 1) -> np.ndarray:
    """
    draw random samples from normal distribution

    :param mu: mean of the distribution
    :param sigma: standard deviation of the distribution
    :param size: number of random samples tb drawn
    :return: sampled values
    """
    return norm.rvs(loc=mu, scale=sigma, size=size)


def draw_bernoulli(p: float = 0.5, size: int = 1) -> np.ndarray:
    """
    draw random samples from bernoulli distribution

    :param p: success probability
    :param size: number of random samples tb drawn
    :return: sampled values
    """
    return np.random.choice([True, False], p=[p, 1.0 - p], size=size)


def draw_uniformly_random_from_values(values: np.ndarray, size: int = 1) -> np.ndarray:
    """
    draw samples from provided values uniformly at random (with replacement)

    :param values: set of values from which samples are drawn uniformly at random
    :param size: number of random samples tb drawn
    :return: sampled values
    """
    return np.random.choice(values, size=size)
