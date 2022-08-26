import datetime as datetime  # Imports {{{
import math
import os
import pickle
import sys
import warnings
from multiprocessing import Pool, Process

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from arch import arch_model
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
k["pred_cov"] = None
k["nu"] = None
k["rmtind"] = None
dist = "t"
window = 300


class resobj(optimize_mgh):
    def __init__(self, k):
        super().__init__(self)
        self.k = k


class Logger(object):
    def __init__(self, pid):
        self.terminal = sys.stdout
        self.log = open(f"{pid}.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def get_res(useREig, useRMT, l, dist):
    pid = str(os.getpid())
    sys.stdout = open(f"{pid}.out", "w")
    for i in range(len(k) - (window + 1)):
        ktrain = k.iloc[i : i + window]
        if i == 0:
            # op = optimize_mgh(ktrain, useRMT=True, useREig=True, dist_type=dist)
            op = optimize_mgh(
                ktrain, l=l, useREig=useREig, useRMT=useRMT, dist_type=dist
            )
            op.outer_minimizer(-8.9)
        else:
            print(f"Time={i}")
            prevk = op.k.copy()
            lamb = op.final_lamb
            mu = op.mu
            gamma = op.gamma
            ktrain["cov_t"] = prevk.cov_t.iloc[1:].tolist() + [
                ktrain.cov_t.iloc[-1]
            ]
            op.k = ktrain
            # op = optimize_mgh(
            #    ktrain,
            #    mu=mu,
            #    gamma=gamma,
            #    useREig=True,
            #    useRMT=True,
            #    dist_type=dist,
            # )
            # op = optimize_mgh(ktrain, l=3, dist_type=dist)
            op.outer_minimizer(lamb)

        k["pred_cov"].loc[i + window + 1] = op.k.cov_t.iloc[-1]
        k["nu"].loc[i + window + 1] = -2 * op.final_lamb
        if useRMT != None:
            k["rmtind"].loc[i + window + 1] = op.rmtindex
        k["mu"].loc[i + window + 1] = op.mu
        objres = resobj(k)

        with open("f{pid}_res.pkl", "wb") as outp:
            pickle.dump(objres, outp, pickle.HIGHEST_PROTOCOL)


p1 = (True, True, None, "t")
p2 = (True, False, None, "t")
p3 = (False, True, 3, "t")
p4 = (False, False, 3, "t")
p5 = (True, True, None, "skew")
p6 = (True, False, None, "skew")
p7 = (False, True, 3, "skew")
# p8 = (False, False, 3, "skew")

with Pool(8) as pool:

    pool.starmap(get_res, [p1, p2, p3, p4, p5, p6, p7])

print(k)
k.to_pickle("t_const_l_results.pkl")
