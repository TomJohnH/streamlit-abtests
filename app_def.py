import numpy as np
import math
from scipy import stats


##################################################
#
#                functions declaration
#
##################################################


def detect(success, population):

    # https://blog.allegro.tech/2019/08/ab-testing-calculating-required-sample-size.html

    # power changed to 80%
    # test power - chance to obtain true postive results - finding a difference when it really exists
    # signifcance level - risk of obtaining false positive results - finding a non existisng difference

    # base conversion rate
    mu = success / population

    # Minimum Detactable Effect
    MDE = math.sqrt(
        2
        * (
            (
                (round(z_score(0.05), 2) + round(z_score((1 - 0.8) * 2), 2)) ** 2
            )  # here we are using *2 due to definition of z_score
            * mu
            * (1 - mu)
        )
        / ((mu ** 2) * population)
    )

    return MDE, mu


def diffprop(obs):
    """
    `obs` must be a 2x2 numpy array.

    Returns:
    delta
        The difference in proportions
    ci
        The Wald 95% confidence interval for delta
    corrected_ci
        Yates continuity correction for the 95% confidence interval of delta.
    """
    n1, n2 = obs.sum(axis=1)
    prop1 = obs[0, 0] / n1
    prop2 = obs[1, 0] / n2
    delta = prop1 - prop2

    # Wald 95% confidence interval for delta
    se = np.sqrt(prop1 * (1 - prop1) / n1 + prop2 * (1 - prop2) / n2)
    ci = (delta - 1.96 * se, delta + 1.96 * se)

    # Yates continuity correction for confidence interval of delta
    correction = 0.5 * (1 / n1 + 1 / n2)
    corrected_ci = (ci[0] - correction, ci[1] + correction)

    return delta, ci, corrected_ci


def z_score(alpha):
    # Calculate the z-score corresponding to the given significance level
    z = stats.norm.ppf(1 - alpha / 2)

    return z
