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
from scipy.special import kv
from scipy.stats import multivariate_normal, norm, t
from sympy import *

from gig import moment_bessel

warnings.filterwarnings("ignore")

os.chdir("/home/sahil/diss")
k = pd.read_pickle("returns_pca.pkl")

ret_col = [i for i in k.columns if "_logret" in i]

d = 29


def multskewt(chi, psi=0, mu=np.array([[0]] * d), gamma=np.array([[0]] * d)):

    lamb = -chi / 2 - d / 2

    chi_star = lambda cov, x: chi + (x - mu).T * (cov ** -1) * (x - mu)
    psi_star = lambda cov, x: psi + gamma.T * (cov ** -1) * gamma

    k["chistar_t"] = k.apply(
        lambda row: chi_star(row["cov_t"], np.matrix(row[ret_col]).T)
    )
    k["psistar_t"] = k.apply(
        lambda row: psi_star(row["cov_t"], np.matrix(row[ret_col])).T
    )
    k["gt"], k["gtinv"] = k.apply(
        lambda row: moment_bessel(lamb, row["chistar_t"], row["phistar_t"])
    )
    return k


k = multskewt(5.6)
print(k)
