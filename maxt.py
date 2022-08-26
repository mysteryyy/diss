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


def multivariate_t_distribution(x, mu, Sigma, df, d):
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
        * pow(df * pi, 1.0 * d / 2)
        * pow(np.linalg.det(Sigma), 1.0 / 2)
        * pow(
            1
            + (1.0 / df)
            * np.dot(np.dot((x - mu), np.linalg.inv(Sigma)), (x - mu)),
            1.0 * (d + df) / 2,
        )
    )
    d = 1.0 * Num / Denom
    return d


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
    exp_comp = (x - mu).T * siginv * skew
    denom1 = bessel_comp ** -bessel_index
    denom2 = math.pow((1 + rho / 2), bessel_index)

    pdf = (
        c
        * (kv(bessel_index, bessel_comp) * np.exp(exp_comp))
        / (denom1 * denom2)
    )
    return pdf


iterno = 0

uniskewt = (
    lambda x, df, alpha: 2.0
    * t.pdf(x, df)
    * t.cdf(df + 1, alpha * x * np.sqrt((1 + df) / (x ** 2 + df)))
)


def uni_likelihood_skewt(alpha, beta, omega, y, bound, v):
    def inner_minimizer(params):
        alpha = params[0]
        beta = params[1]
        omega = params[2]
        if alpha + beta > 1:
            return 1e6
        if alpha < 0 or beta < 0 or omega < 0:
            return 1e6
        model = arch_model(
            pd.Series(y),
            mean="Zero",
            vol="GARCH",
            p=1,
            o=0,
            q=1,
            dist="StudentsT",
        )
        try:
            res1 = model.fix([omega, alpha, beta, v])
        except Exception as e:
            print(e)

        if math.isnan(res1.loglikelihood):
            return 1e6

        return -res1.loglikelihood

    while True:
        try:
            res = minimize(
                inner_minimizer,
                [alpha] + [beta] + [omega],
                method="slsqp",
                bounds=bound,
                tol=1e-6,
            )
            break
        except Exception as e:
            print(e)
            change = lambda x: x + np.random.uniform(0, 1)
            omega = change(omega)
            alpha = change(alpha)
            beta = change(beta)

    return res


def likelihood_t(y, alpha, beta, omega, v, init_vol):
    # Dimension of data
    d = y.shape[1]

    def logger(func):
        def inner(*args, **kwargs):
            global iterno
            iterno = iterno + 1
            params = args[0]
            print(f"df value: {params[0]}")
            val = func(*args, **kwargs)
            print(f"loss value:{val}")

        return inner

    new_params = []

    # @logger
    def outer_minimizer(params):
        v = params[0]
        loss = []
        for j in range(d):
            print(f"v={v}")
            alpha = 0.1
            beta = 0.8
            omega = 0.5

            bound = ((0.005, 1),) * 2 + ((0.005, 0.95),)
            bound = bound
            ycomp = y[:][:, j]
            res = uni_likelihood_skewt(alpha, beta, omega, ycomp, bound, v)
            loss.append(res.fun)
            print(res.x)
            new_params.append(res.x)
        totalloss = np.array(loss).sum()
        if math.isnan(totalloss):
            return 1e9
        print(f"{v} loss = {totalloss}")
        return totalloss

    res = minimize(
        outer_minimizer,
        [4.23],
        method="slsqp",
        bounds=((2.01, 100.0),),
        tol=1e-6,
    )

    return res.fun


# Column names of principal components
pcols = ["p" + str(i + 1) for i in range(29)]
k = pd.read_pickle("returns_pca.pkl")
ktrain = k[k.Date < datetime.date(2019, 6, 1)]
ktest = k[k.Date > datetime.date(2019, 6, 1)]

res = likelihood_t(
    np.array(ktrain[pcols]),
    [0.1] * 29,
    [0.8] * 29,
    [0.03] * 29,
    5,
    [2.1] * 29,
)
# A = [45, 37, 42, 35, 39]
# B = [38, 31, 26, 28, 33]
# C = [10, 15, 17, 21, 12]
#
# data = np.array([A, B, C])
#
# covMatrix = np.cov(data, bias=True)
# sigma = covMatrix
# mu = np.matrix([0, 0, 0]).T
# x = np.random.multivariate_normal(
#    [0, 0, 0],
#    sigma,
#    1,
# ).T
# skew = np.matrix([0.01, 0.02, 0.01]).T
# v = 5.1
# print(multskewt(x, mu, skew, sigma, v))
