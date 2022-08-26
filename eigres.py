import datetime as datetime  # Imports {{{
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
lenk = int(len(k) / 2)
ktrain = k.iloc[:lenk]
ktest = k.iloc[lenk:]
ret_col = [i for i in ktrain.columns if "_logret" in i]
eigval, evec = getPCA(ktrain[ret_col].cov())
eigval, evec1 = getPCA(ktest[ret_col].cov())
