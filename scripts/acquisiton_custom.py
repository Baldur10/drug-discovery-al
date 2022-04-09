from typing import Tuple

import numpy as np
from scipy.stats import norm
from scipy.special import ndtr
from sklearn.exceptions import NotFittedError

from modAL.utils.selection import multi_argmax
from modAL.utils.data import modALinput
from modAL.models.base import BaseLearner

def PI(mean, std, max_val, tradeoff):
    return ndtr((mean - max_val - tradeoff)/std)

def optimizer_PI(optimizer: BaseLearner, X: modALinput, tradeoff: float = 0) -> np.ndarray:
    """
    Probability of improvement acquisition function for Bayesian optimization.
    Args:
        optimizer: The :class:`~modAL.models.BayesianOptimizer` object for which the utility is to be calculated.
        X: The samples for which the probability of improvement is to be calculated.
        tradeoff: Value controlling the tradeoff parameter.
    Returns:
        Probability of improvement utility score.
    """
    try:
        mean = optimizer.predict(X)
        mean, std = mean.reshape(-1, ), std.reshape(-1, )
    except NotFittedError:
        mean, std = np.zeros(shape=(X.shape[0], 1)), np.ones(shape=(X.shape[0], 1))

    return PI(mean, std, optimizer.y_max, tradeoff)
