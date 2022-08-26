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
from sklearn.decomposition import PCA

from gig import (
    lnw,
    log_gamma,
    loglike_ig,
    moment_bessel,
    moment_gamma_x,
    moment_gamma_xinv,
)
from skewtpdf import multskewtpdf

warnings.filterwarnings("ignore")

os.chdir("/home/sahil/diss")
k = pd.read_pickle("returns_pca.pkl")

ret_col = [i for i in k.columns if "_logret" in i]

d = 29
k = k.iloc[1:]


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

        return w * np.diag(pvar) * w.T

    def gigparams(self, lamb, chi, psi=0):
        lamb = lamb - self.d / 2
        k = self.k.copy()
        k["chistar_t"] = k.apply(
            lambda row: np.float(
                self.calc_chi_star(
                    chi, row["cov_t"], np.matrix(row[ret_col]).T
                )
            ),
            axis=1,
        )
        k["gt"] = k.apply(
            lambda row: moment_gamma_x(-lamb, row["chistar_t"]),
            axis=1,
        )
        k["ginvt"] = k.apply(
            lambda row: moment_gamma_xinv(-lamb, row["chistar_t"]),
            axis=1,
        )
        k["lngt"] = k.apply(
            lambda row: log_gamma(
                -lamb,
                row["chistar_t"],
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

    def loglike_gig(self, lamb, psi=1e-10):
        length = len(self.k)
        nu = -2 * (-1 - np.exp(lamb))
        print(f"nu={nu}")
        g = self.k["gt"]
        lnt = self.k["lngt"]
        ginv = self.k["ginvt"]
        term1 = -length * nu * np.log(nu / 2 - 1) / 2
        term2 = (nu / 2 + 1) * lnt.sum() + (nu / 2 - 1) * ginv.sum()
        term3 = length * np.log(gamma(nu / 2))
        totallog = term1 + term2 + term3
        return totallog / len(self.k)

    def eigdecomp(self, x):
        cov = (x.T * x) / (len(x))
        w, v = np.linalg.eigh(cov)
        return w, v

    def multskewt(self, lamb, psi=1e-10):

        pcomp_cols = [f"p{i+1}" for i in range(self.d)]
        pcolsvar = [f"p{i+1}_vol" for i in range(29)]
        # initializing lambda parameter
        chi = -2 * (lamb + 1)
        k = self.gigparams(lamb, chi, psi=psi)
        self.k = k
        # performing required regression

        reg_res = self.ols(k)
        reg_res.columns = self.ret_cols
        # Storing values of mean and skewness
        self.mu = np.matrix(reg_res.loc["const"]).T
        # self.mu = np.matrix([1e-5]).T
        self.skew = np.matrix(reg_res.loc["gt"]).T
        # self.skew = np.matrix([1e-5]).T

        # Storing column names of errror columns
        err_col = [f"err_{i}" for i in self.ret_cols]
        err_col_stand = [f"err_{i}_stand" for i in self.ret_cols]
        df = pd.DataFrame()

        # calculating errors
        err_log = 0

        for i in self.ret_cols:
            err_col_name = f"err_{i}"
            # Extracting the mean from the returns
            df[err_col_name] = k[i] - reg_res[i].loc["const"]
            err_log_col = df[err_col_name] ** 2
            err_log = err_log + err_log_col.sum()
        err_log = -0.5 * err_log
        for i, j in zip(err_col, err_col_stand):
            df[j] = df[i] * k["ginvt"] ** 0.5
        #  Spectral Decomposition
        v, w = self.eigdecomp(np.matrix(df[err_col_stand]))
        self.eigvec = w
        self.eigval = v

        # Extracting Principal Components
        df[pcomp_cols] = np.array(np.matrix(df[err_col]) * w)
        df["ginvt"] = k["ginvt"]
        df["gt"] = k["gt"]

        # for i in pcomp_cols:
        #    df[i] = df[i] * df["ginvt"] ** 0.5
        plog = 0
        for i, j in zip(pcomp_cols, self.ret_cols):
            res = arch_model(df[i], mean="Zero", vol="GARCH", p=1, o=0, q=1)
            mod = res.fit()
            vol_col = f"{i}_vol"
            k[vol_col] = mod.conditional_volatility ** 2
            df[vol_col] = k[vol_col]
            # loglike = (
            #    np.log(2 * np.pi * k["gt"])
            #    + np.log(df[vol_col])
            #    + k["ginvt"] * (df[i] ** 2 / df[vol_col])
            #    - df[f"err_{j}"] ** 2 * k["ginvt"]
            # )
            loglike = mod.loglikelihood
            plog = plog - loglike

        k["cov_t"] = k[pcolsvar].apply(
            lambda row: self.cov_est(row, w), axis=1
        )
        # k["cov_t"] = k["cov_t"].apply(
        #    lambda x: math.pow(c, 1 / self.d)
        #    * x
        #    / (np.linalg.det(x)) ** (1 / self.d)
        # )

        self.k = k
        totallog = plog
        return totallog / (len(self.k))


#    def loglike(lamb,chi,psi):


def outer_minimizer(lamb, k):
    prev_loglike = 0
    mu = np.matrix(k[ret_col].mean()).T
    gamma = np.matrix(k[ret_col].skew()).T
    op = optimize(mu, gamma, k)

    def inner_minimizer(params):
        lamb = params[0]
        giglog = op.loglike_gig(lamb)
        print(f"iglog ={giglog}")
        totlog = -4 * normlog + giglog
        return totlog

    it = 0
    mu = op.mu.tolist()[0]
    chi = -2 * (lamb + 1)
    loglike_t = lambda k, mu, nu: k.apply(
        lambda row: np.log(
            multivariate_t_distribution(row[ret_col], mu, row["cov_t"], nu, 29)
        ),
        axis=1,
    ).mean()

    prev_loglike = loglike_t(k, mu, chi)
    print(f"Begining Log likelihood ={prev_loglike}")
    while True:
        # try:
        it = it + 1
        print(f"iter = {it}")
        # try:
        normlog = op.multskewt(lamb)
        print(f"normlog={normlog}")
        # except Exception as e:
        normlog = 1000
        lamb = lamb - np.random.uniform(0, 1) * 3
        if lamb == -1.0:
            lamb = lamb - np.random.uniform(0, 1) * 3
        res = minimize(
            inner_minimizer,
            [np.log(-1 - lamb)],
            method="nelder-mead",
            bounds=((-200.0, 0.001),),
            tol=1e-6,
        )
        lamb = -1 - np.exp(res.x[0])
        chi = -2 * (lamb + 1)
        mu = op.mu.flatten().tolist()[0]
        curr_loglike = loglike_t(op.k, mu, chi)
        # for i in range(len(k)):
        #    x = k[ret_col].iloc[i]
        #    mu = op.mu.tolist()[0]
        #    cov = k.cov_t.iloc[i]
        #    pdf = multivariate_t_distribution(x, mu, cov, chi, 29)
        #    lpdf = np.log(pdf)
        #    totlog = totlog + lpdf
        # curr_loglike = totlog / len(k)
        print(f"Current Loglikelihood={curr_loglike}")
        print(f"Current value of lamb={lamb}")

        diff = curr_loglike - prev_loglike
        print(f"Current Difference = {diff}")
        prev_loglike = curr_loglike
        if abs(diff) < 0.000001:
            break
    return op, lamb, curr_loglike


ktrain = k[k.Date < datetime.date(2019, 6, 1)]
# ktrain = ktrain.iloc[200:500]
ret_cols = [i for i in k.columns if "_logret" in i]
c = np.linalg.det(ktrain[ret_cols].cov())
res, lamb, curr_loglike = outer_minimizer(-14.5, ktrain)
print(lamb)
