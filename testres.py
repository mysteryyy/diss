import datetime as datetime  # Imports {{{
import math
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from arch import arch_model
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.hierarchical_portfolio import *
from scipy.optimize import minimize
from scipy.special import gamma, kv
from scipy.stats import multivariate_normal, norm, t
from sklearn.decomposition import PCA

from gig import (
    lnw,
    log_gamma,
    loglike_ig,
    moment_bessel,
    moment_gamma_x,
    moment_gamma_xinv,
)
from maxskewt4 import optimize_mgh
from random_corr import *
from tpdfs import multivariate_t_distribution, multskewtpdf

warnings.filterwarnings("ignore")
# }}}
os.chdir("/home/sahil/diss")
k = pd.read_pickle("returns_pca.pkl")
k = k.iloc[1:]

k = pd.read_pickle("t_const_l_results.pkl")
k1 = pd.read_pickle("t_results.pkl")
ret_col = [i for i in k.columns if "_logret" in i]


def detonedcorr(corr, alpha=0):
    # Remove noise from corr through targeted shrinkage
    eVal, eVec = getPCA(corr)
    eval1, evec1 = eVal[0][0], eVec[:, :1]
    corr1 = corr - np.dot(evec1, eval1).dot(evec1.T)
    corr1 = cov2corr(corr1)
    return corr1


def optim(x, method=None):
    if method == "hrp":
        hrp = HRPOpt(cov_matrix=pd.DataFrame(x))
        hrp.optimize()
        return pd.Series(hrp.clean_weights()).tolist()
    ef = EfficientFrontier([0] * 29, x)
    ef.min_volatility()
    w = pd.Series(ef.clean_weights()).tolist()
    return w


def performance(ret, w):
    rets = np.diagonal(ret * w.T)
    mu = rets.mean()
    var = rets.var()
    print(f"variance={var}")
    return mu * 252 / ((var * 252) ** 0.5)


def get_sharpe(k, method):
    k = k.drop("rmtind", axis=1).dropna()
    k["weights"] = k.pred_cov.apply(lambda x: optim(x))
    # Calc dynamic cirrelations
    k["corr"] = k.pred_cov.apply(lambda x: cov2corr(x))
    # Calculate reduced correlation removing the market component
    k["red_corr"] = k["corr"].apply(lambda x: detonedcorr(x))
    k["hrp_weights"] = k.red_corr.apply(lambda x: optim(x, method="hrp"))
    k.to_pickle(f"optim_{method}.pkl")
    # k["hrp_weights_normal"] = k.red_corr.apply(lambda x: optim(x, method="hrp"))
    print("normal performance")
    ret = np.matrix(k[ret_col])
    w = np.matrix(k.weights.tolist())
    print(performance(ret, w))
    print("hrp performance")
    w = np.matrix(k.hrp_weights.tolist())
    print(performance(ret, w))


print(get_sharpe(k, "const_l_weights"))
print(get_sharpe(k1, "rmt_weights"))
