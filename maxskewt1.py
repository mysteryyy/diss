import datetime as datetime
import math
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from arch import arch_model
from pypfopt import risk_models
from scipy.optimize import minimize
from scipy.special import kv
from scipy.stats import multivariate_normal, norm, t
from sklearn.decomposition import PCA
from sympy import *

from gig import digamm, lnw, log_gamma, loglike_ig, moment_bessel, moment_gamma
from skewtpdf import multskewtpdf

warnings.filterwarnings("ignore")

os.chdir("/home/sahil/diss")
k = pd.read_pickle("returns_pca.pkl")

ret_col = [i for i in k.columns if "_logret" in i]

d = 29
k = k.iloc[1:]


class optimize:
    def __init__(self, mu, gamma, k):
        self.d = mu.shape[0]
        self.mu = mu
        self.gamma = gamma
        self.k = k
        self.ret_cols = [i for i in self.k.columns if "_logret" in i]

    def calc_chi_star(self, chi, cov, x):
        chi_star = chi + (x - self.mu).T * (cov ** -1) * (x - self.mu)
        return chi_star

    def calc_psi_star(self, cov, x, psi=0):
        return psi + self.gamma.T * (cov ** -1) * self.gamma

    def ols(self, k):
        x = k["gt"]

        x = sm.add_constant(x)
        x = x.astype("float")
        res = sm.OLS(self.k[self.ret_cols], x).fit().params
        return res

    def cov_est(self, pvar, w):

        return w.T * np.diag(pvar) * w

    def gigparams(self, lamb, chi, psi=0):
        k = self.k.copy()
        k["chistar_t"] = k.apply(
            lambda row: self.calc_chi_star(
                chi, row["cov_t"], np.matrix(row[ret_col]).T
            ),
            axis=1,
        )
        k["psistar_t"] = k.apply(
            lambda row: self.calc_psi_star(
                row["cov_t"], np.matrix(row[ret_col]).T
            ),
            axis=1,
        )

        k["gt"] = k.apply(
            lambda row: moment_bessel(
                -lamb, row["chistar_t"], row["psistar_t"], 1
            ),
            axis=1,
        )
        k["ginvt"] = k.apply(
            lambda row: moment_bessel(
                -lamb, row["chistar_t"], row["psistar_t"], -1
            ),
            axis=1,
        )
        k["lngt"] = k.apply(
            lambda row: log_gamma(
                -lamb,
                np.float(row["chistar_t"]),
            ),
            axis=1,
        )
        return k

    def loglike(self, chi):
        self.k["logt"] = self.k.apply(
            lambda row: np.log(
                np.float(
                    multskewtpdf(
                        np.matrix(row[self.ret_cols]).T,
                        self.mu,
                        self.skew,
                        row["cov_t"],
                        chi,
                    )
                )
            ),
            axis=1,
        )
        return self.k.logt.mean()

    def loglike_gig(self, chi, psi=1e-10):
        length = len(self.k)
        lamb = -chi / 2
        g = self.k["gt"]
        lnt = np.log(self.k["gt"])
        ginv = self.k["ginvt"]
        logval = (
            -digamm(chi / 2)
            + np.log(chi / 2)
            + 1
            - (ginv.sum() + lnt.sum()) / length
        )
        return logval

    def multskewt(self, chi, c, psi=1e-10):

        pcomp_cols = [f"p{i+1}" for i in range(self.d)]
        pcolsvar = [f"p{i+1}_vol" for i in range(29)]
        # initializing lambda parameter
        lamb = -chi / 2 - self.d / 2
        self.k["cov_t"] = self.k["cov_t"].apply(
            lambda x: math.pow(c, 1 / self.d)
            * x
            / math.pow(np.linalg.det(x), 1 / self.d)
        )

        k = self.gigparams(lamb, chi, psi=psi)
        # Store the determinant of sample covariance matrix for later use

        # performing required regression

        reg_res = self.ols(k)
        reg_res.columns = self.ret_cols
        # Storing values of mean and skewness
        self.mu = np.matrix(reg_res.loc["const"]).T
        # self.mu = np.matrix([1e-5]).T
        self.skew = np.matrix(reg_res.loc["gt"]).T
        # self.skew = np.matrix([1e-5]).T
        self.k = k

        # Storing column names of errror columns
        err_col = [f"err_{i}" for i in self.ret_cols]
        df = pd.DataFrame()

        # calculating errors
        err_log = 0

        for i in self.ret_cols:
            err_col_name = f"err_{i}"
            df[err_col_name] = (
                k[i] - reg_res[i].loc["gt"] * k["gt"] - reg_res[i].loc["const"]
            )
            err_log_col = df[err_col_name] ** 2 * k["ginvt"]
            err_log = err_log + err_log_col.sum()
        err_log = -0.5 * err_log
        # Extracting the principal components
        pca = PCA(n_components=self.d)
        pca = pca.fit(df[err_col])
        w = np.matrix(pca.components_)
        lamb = pca.explained_variance_
        df[pcomp_cols] = pca.transform(df[err_col])
        df["ginvt"] = k["ginvt"]
        df["gt"] = k["gt"]

        for i in pcomp_cols:
            df[i] = df[i] * df["ginvt"] ** 0.5
        plog = 0
        for i, j in zip(pcomp_cols, self.ret_cols):
            res = arch_model(df[i], mean="Zero", vol="GARCH", p=1, o=0, q=1)
            mod = res.fit()
            vol_col = f"{i}_vol"
            k[vol_col] = mod.conditional_volatility ** 2
            df[vol_col] = k[vol_col]
            loglike = (
                np.log(2 * np.pi * k["gt"])
                + np.log(df[vol_col])
                + k["ginvt"] * (df[i] ** 2 / df[vol_col])
                - df[f"err_{j}"] ** 2 * k["ginvt"]
            )
            plog = plog + loglike.sum()

        k["cov_t"] = k[pcolsvar].apply(
            lambda row: self.cov_est(row, w), axis=1
        )
        k["cov_t"] = k["cov_t"].apply(
            lambda x: math.pow(c, 1 / self.d)
            * x
            / (np.linalg.det(x)) ** (1 / self.d)
        )
        self.k = k
        totallog = err_log - 0.5 * plog
        return totallog / len(self.k)


#    def loglike(lamb,chi,psi):


def outer_minimizer(chi, k):
    prev_loglike = 0
    mu = np.array([[0]] * 29)
    gamma = np.array([[-0.01]] * 29)
    op = optimize(mu, gamma, k)

    def inner_minimizer(params):
        chi = params[0]
        giglog = op.loglike_gig(chi)
        print(f"iglog ={giglog}")
        totlog = normlog + giglog
        return -totlog

    it = 0
    curr_loglike = 0
    while True:
        # try:
        it = it + 1
        print(f"iter = {it}")
        # try:
        normlog = op.multskewt(chi, c)
        print(f"normlog={normlog}")
        # except Exception as e:
        # print(e)
        chi = chi + np.random.uniform()
        res = minimize(
            inner_minimizer,
            [chi],
            method="slsqp",
            bounds=((4.001, 200.0),),
            tol=1e-6,
        )
        curr_loglike = -res.fun
        chi = res.x[0]
        print(f"Current Loglikelihood={curr_loglike}")
        print(f"Current value of chi={chi}")

        diff = curr_loglike - prev_loglike
        print(f"Current Difference = {diff}")
        prev_loglike = curr_loglike
        if abs(diff) < 0.001:
            break
    return op, chi, curr_loglike


ktrain = k[k.Date < datetime.date(2019, 6, 1)]
# ktrain = k.iloc[200:1201]
ret_cols = [i for i in k.columns if "_logret" in i]
c = np.linalg.det(ktrain[ret_cols].cov())
# ktrain["cov_t"] = ktrain.apply(
#    lambda row: np.matrix(risk_models.cov_to_corr(row["cov_t"])), axis=1
# )
# ktrain["cov_t"] = ktrain["cov_t"].apply(lambda x: x / np.linalg.det(x))
# ktrain = k.iloc[0:1000]
res, chi, curr_loglike = outer_minimizer(5.7, ktrain)
print(chi)
