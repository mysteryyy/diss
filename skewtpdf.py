import datetime as datetime
import math
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from arch import arch_model
from scipy.optimize import minimize
from scipy.special import gamma, kv
from scipy.stats import multivariate_normal, norm, t
from sympy import *

warnings.filterwarnings("ignore")

os.chdir("/home/sahil/diss")


def normalize_input(dist):
    def wrapper_normalize_input(*args, **kwargs):
        x, mu, skew, sigma, v = args[0], args[1], args[2], args[3], args[4]

        if isinstance(x, (list, tuple, np.ndarray)) == False:
            return dist(
                np.matrix([x]),
                np.matrix([mu]),
                np.matrix([skew]),
                np.matrix([sigma]),
                v,
            )
        else:
            return dist(*args, **kwargs)

    return wrapper_normalize_input


@normalize_input
def multskewtpdf(x, mu, skew, sigma, v):
    sigdet = np.linalg.det(sigma)
    siginv = sigma ** -1
    d = x.shape[0]
    rho = (x - mu).T * sigma ** -1 * (x - mu)
    c = (2 ** (1 - (v + d) / 2)) / (
        gamma(v / 2) * (np.pi * v) ** (d / 2) * sigdet ** 0.5
    )
    bessel_comp = math.sqrt(((v + rho) * (skew.T * siginv * skew)))
    bessel_index = (v + d) / 2
    exp_comp = np.float((x - mu).T * siginv * skew)
    denom1 = bessel_comp ** -bessel_index
    denom2 = math.pow((1 + rho / 2), bessel_index)

    pdf = (
        c
        * (kv(bessel_index, bessel_comp) * np.exp(exp_comp))
        / (denom1 * denom2)
    )
    return pdf


def multivariate_t_distribution(x, mu, Sigma, df, d):  # T distribution pdf{{{
    """
    Multivariate t-student density:
    output:
        the density of the given element
    input:
        x = parameter (d dimensional numpy array or scalar)
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
        d: dimension
    """
    Num = gamma(1.0 * (d + df) / 2)
    Denom = (
        gamma(1.0 * df / 2)
        * pow(df * np.pi, 1.0 * d / 2)
        * pow(np.linalg.det(Sigma), 1.0 / 2)
        * pow(
            1
            + (1.0 / df)
            * np.float(
                np.dot(np.dot((x - mu), np.linalg.inv(Sigma)), (x - mu))
            ),
            1.0 * (d + df) / 2,
        )
    )
    d = 1.0 * Num / Denom
    return d


# }}}
