__author__ = 'arenduchintala'
import numpy as np
import copy
from math import sqrt


def cosine_sim(v1, v2):
    assert len(v1) == len(v2)
    assert isinstance(v1, np.ndarray)
    assert isinstance(v2, np.ndarray)
    dot = 0.0
    v1_sq = 0.0
    v2_sq = 0.0
    for i in xrange(len(v1)):
        dot += v1[i] * v2[i]
        v1_sq += v1[i] ** 2.0
        v2_sq += v2[i] ** 2.0
    denom = (sqrt(v1_sq) * sqrt(v2_sq))
    if denom > 0.0:
        return dot / denom
    else:
        return float('inf')


def gradient_checking(theta, eps, cost_function, data):
    f_approx = np.zeros(np.shape(theta))
    for i, t in enumerate(theta):
        theta_plus = copy.deepcopy(theta)
        theta_minus = copy.deepcopy(theta)
        theta_plus[i] = theta[i] + eps
        theta_minus[i] = theta[i] - eps
        f_approx[i] = (cost_function(theta_plus, data) - cost_function(theta_minus, data)) / (2 * eps)
    return f_approx