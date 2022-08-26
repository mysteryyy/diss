import datetime as datetime
import math
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from arch import arch_model
from scipy.misc import derivative
from scipy.optimize import minimize
from scipy.special import digamma, gamma, kv
from scipy.stats import multivariate_normal, norm, t

warnings.filterwarnings("ignore")


def limit_bessel(lamb, x):
    return gamma(-lamb) * 2 ** (-lamb - 1) * x ** lamb


def moment_bessel(lamb, chi, psi, moment):
    chi = np.float(chi)
    psi = np.float(psi)
    eta = (chi / psi) ** (moment / 2)
    omega = np.sqrt(chi * psi)
    if omega > 0.001:
        kmod = lambda x: kv(x, omega)
    else:
        kmod = lambda x: limit_bessel(x, omega)

    ex = eta * kmod(lamb + moment) / kmod(lamb)
    return ex


def skewt_solve(v, eta, delta):
    return -digamma(v / 2) + np.log(v / 2) + 1 - eta - delta


def lnw(lamb, chi, psi, moment=0):
    h = 0.0001
    x1 = moment + h
    x = moment
    f1 = moment_bessel(lamb, chi, psi, x1)
    f = moment_bessel(lamb, chi, psi, x)
    kmod = lambda x: kv(x, omega)

    omega = np.sqrt(chi * psi)
    first_term = 0.5 * np.log(chi / psi) + derivative(kmod, lamb, dx=1e-6)
    return first_term


def moment_gamma_x(alpha, beta):
    return beta / (alpha - 1)


def moment_gamma_xinv(alpha, beta):
    return alpha / (beta)


def digamm(x):
    return np.log(x) - 1 / (2 * x)


def log_gamma(alpha, beta):
    return np.log(beta) - digamma(alpha)


def loglike_ig(lnx, invx, alpha, beta):
    return (
        alpha * np.log(beta)
        - np.log(gamma(alpha))
        - (alpha + 1) * lnx
        - beta * invx
    )
