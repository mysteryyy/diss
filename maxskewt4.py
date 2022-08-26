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
from random_corr import *
from tpdfs import multivariate_t_distribution, multskewtpdf

warnings.filterwarnings("ignore")
# }}}
os.chdir("/home/sahil/diss")
k = pd.read_pickle("returns_pca.pkl")
k = k.iloc[1:]


class optimize_mgh:
    def __init__(
        self,
        k,
        mu=None,
        gamma=None,
        useRMT=False,
        l=None,
        useREig=False,
        dist_type="t",
    ):
        self.k = k
        self.ret_cols = [i for i in self.k.columns if "_logret" in i]
        self.d = len(self.ret_cols)
        self.mu = mu
        self.gamma = gamma
        self.useRMT = useRMT
        self.useREig = useREig
        self.l = l
        self.dist_type = dist_type
        if useRMT == False and l == None:
            self.l = self.d

    def calc_chi_star(self, chi, cov, x):  #  For estimating g and g^-1{{{
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

    def gigparams(self, lamb, chi, psi=0):
        lamb = lamb - self.d / 2
        k = self.k.copy()
        k["chistar_t"] = k.apply(
            lambda row: np.float(
                self.calc_chi_star(
                    chi, row["cov_t"], np.matrix(row[self.ret_cols]).T
                )
            ),
            axis=1,
        )
        if self.dist_type == "t":

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
        else:
            k["psistar_t"] = k.apply(
                lambda row: self.calc_psi_star(
                    row["cov_t"], np.matrix(row[self.ret_cols]).T
                ),
                axis=1,
            )

            k["gt"] = k.apply(
                lambda row: moment_bessel(
                    lamb, row["chistar_t"], row["psistar_t"], 1
                ),
                axis=1,
            )
            k["ginvt"] = k.apply(
                lambda row: moment_bessel(
                    lamb, row["chistar_t"], row["psistar_t"], -1
                ),
                axis=1,
            )
            # k["lngt"] = k.apply(
            #    lambda row: lnw(lamb, row["chistar_t"], row["psistar_t"]),
            #    axis=1,
            # )
            k["lngt"] = np.log(k["gt"])

        return k  # }}}

    def loglike(self, chi):  # loglike for skew t distribution {{{
        self.k["logt"] = self.k.apply(
            lambda row: np.log(
                np.float(
                    multskewtpdf(
                        np.matrix(row[self.ret_cols]).T,
                        self.mu,
                        self.gamma,
                        row["cov_t"],
                        chi,
                    )
                )
            ),
            axis=1,
        )
        return self.k.logt.mean()

    # }}}
    def cov_est(self, pvar, w, index):
        evals = self.varpcomp

        if index < self.d - 1:
            if self.useREig == True:
                ev_mean = np.array(evals[index + 1 :]).mean()
                pvar = pvar[0 : index + 1].tolist() + [ev_mean] * (
                    self.d - index - 1
                )
            else:
                pvar = pvar[0 : index + 1].tolist() + evals[index + 1 :]

        return w * np.diag(pvar) * w.T

    def loglike_gig(self, lamb, psi=1e-10):  # loglike for nu {{{
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

    # }}}
    def eigd(self, x):
        cov = (x.T * x) / len(x)
        v, w = np.linalg.eigh(cov)
        return w, v

    def multskewt(self, lamb, psi=1e-10):

        pcomp_cols = [f"p{i+1}" for i in range(self.d)]
        pcolsvar = [f"p{i+1}_vol" for i in range(29)]
        # initializing lambda parameter
        chi = -2 * (lamb + 1)
        k = self.gigparams(lamb, chi, psi=psi)
        self.k = k
        # Performing required regression

        reg_res = self.ols(k)
        reg_res.columns = self.ret_cols
        # Storing values of mean and skewness
        self.mu = np.matrix(reg_res.loc["const"]).T
        # self.mu = np.matrix([1e-5]).T
        self.gamma = np.matrix(reg_res.loc["gt"]).T
        # self.skew = np.matrix([1e-5]).T

        # Storing column names of errror columns
        err_col = [f"err_{i}" for i in self.ret_cols]
        err_col_stand = [f"err_{i}_stand" for i in self.ret_cols]
        df = pd.DataFrame()

        # calculating errors
        err_log = 0

        for i in self.ret_cols:
            err_col_name = f"err_{i}"
            # Demean the returns
            if self.dist_type == "skew":
                df[err_col_name] = (
                    k[i]
                    - reg_res[i].loc["gt"] * k["gt"]
                    - reg_res[i].loc["const"]
                )
            elif self.dist_type == "t":
                df[err_col_name] = k[i] - reg_res[i].loc["const"]
            else:
                raise Exception("Wrong distribution specified")

            err_log_col = df[err_col_name] ** 2
            err_log = err_log + err_log_col.sum()

        # Extracting Standardized error
        for i, j in zip(err_col, err_col_stand):
            df[j] = df[i] * k["ginvt"] ** 0.5
        # Performing Eigendecomposition
        length = len(k)
        x = np.matrix(df[err_col_stand])
        cov1 = (x.T * x) * len(x) ** -1
        eVal0, evec = getPCA(cov1)

        # Calculating Eigenvalues that are significant if RMT is used
        if self.useRMT == True:
            x = np.matrix(k[self.ret_cols])
            q = x.shape[0] / float(x.shape[1])
            eMax0, var0 = findMaxEval(np.diag(eVal0), q=q, bWidth=0.01)
            nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
            index = nFacts0 - 1
            self.rmtindex = index
        # In case some constant is specified
        elif self.l != None:
            index = self.l - 1
        else:
            # Case where all principal components are used
            index = len(self.d) - 1
        # Store eigenvalues and eigenvecs for further use
        self.eigvec = evec
        self.eigval = eVal0
        err_log = -0.5 * err_log
        df[pcomp_cols] = np.array(np.matrix(df[err_col]) * evec)
        df["ginvt"] = k["ginvt"]
        df["gt"] = k["gt"]

        # for i in pcomp_cols:
        #    df[i] = df[i] * df["ginvt"] ** 0.5
        plog = 0
        # Storing Unconditional Variances of principal components
        self.varpcomp = []
        self.garchobjs = []
        it = 0
        for i, j in zip(pcomp_cols, self.ret_cols):
            vol_col = f"{i}_vol"
            if it <= index:
                res = arch_model(
                    df[i], mean="Zero", vol="GARCH", p=1, o=0, q=1
                )
                mod = res.fit(
                    starting_values=np.array([0.02, 0.8, 0.15]), disp="off"
                )
                k[vol_col] = mod.conditional_volatility ** 2
            it = it + 1
            self.varpcomp.append(df[i].var())
            df[vol_col] = k[vol_col]
            loglike = mod.loglikelihood
            plog = plog - loglike

        k["cov_t"] = k[pcolsvar].apply(
            lambda row: self.cov_est(row, evec, index), axis=1
        )
        # k["cov_t"] = k["cov_t"].apply(
        #    lambda x: math.pow(c, 1 / self.d)
        #    * x
        #    / (np.linalg.det(x)) ** (1 / self.d)
        # )

        self.k = k
        totallog = plog
        return totallog / (len(self.k))

    def outer_minimizer(self, lamb):

        prev_loglike = 0
        if type(self.mu) != np.matrix and type(self.gamma) != np.matrix:
            self.mu = np.matrix(k[self.ret_cols].mean()).T
            self.gamma = np.matrix(k[self.ret_cols].skew()).T
        # op = optimize(mu, gamma, k)

        def inner_minimizer(params):
            lamb = params[0]
            giglog = self.loglike_gig(lamb)
            print(f"iglog ={giglog}")
            return giglog

        it = 0
        mu = self.mu.tolist()[0]
        chi = -2 * (lamb + 1)
        loglike_t = lambda k, mu, nu: k.apply(
            lambda row: np.log(
                multivariate_t_distribution(
                    row[self.ret_cols], mu, row["cov_t"], nu, self.d
                )
            ),
            axis=1,
        ).mean()

        prev_loglike = (
            self.loglike(chi)
            if self.dist_type == "skew"
            else loglike_t(self.k, mu, chi)
        )
        print(f"Begining Log likelihood ={prev_loglike}")
        prev_diff = 0
        while True:
            # try:
            it = it + 1
            if it > 20:
                self.final_lamb = lamb
                break
            print(f"iter = {it}")
            try:
                normlog = self.multskewt(lamb)
                print(f"normlog={normlog}")
            except ValueError as e:
                print(e)
                lamb = lamb + np.random.uniform(-1, 1)
                continue
            if lamb >= -1.0:
                lamb = lamb - np.random.uniform(0, 1) * 3
            res = minimize(
                inner_minimizer,
                [np.log(-1 - lamb)],
                method="nelder-mead",
                # bounds=((-200.0, 0.001),),
                tol=1e-6,
            )
            lamb = -1 - np.exp(res.x[0])
            # Store optimized lambda value
            self.final_lamb = lamb
            chi = -2 * (lamb + 1)
            mu = self.mu.flatten().tolist()[0]
            if self.dist_type == "skew":
                curr_loglike = self.loglike(chi)
            else:
                curr_loglike = loglike_t(self.k, mu, chi)
            print(f"Current Loglikelihood={curr_loglike}")
            print(f"Current value of lamb={lamb}")

            diff = curr_loglike - prev_loglike
            diffdiff = diff - prev_diff
            if it >= 3:
                if abs(diffdiff) > 1:
                    break
            print(f"Current Difference = {diff}")
            prev_loglike = curr_loglike
            prev_diff = diff
            if abs(diff) < 0.001:
                break


#    def loglike(lamb,chi,psi):


# ktrain = k[k.Date < datetime.date(2019, 6, 1)]
# ktrain = ktrain.iloc[200:800]
# ret_cols = [i for i in k.columns if "_logret" in i]
# op = optimize_mgh(ktrain, useRMT=False, useREig=False, dist_type="skew")
# op.outer_minimizer(-14.5)
# res, lamb, curr_loglike = outer_minimizer(
#    -14.5, ktrain, ret_cols, l=None, useRMT=True, dist_type="skew"
# )
