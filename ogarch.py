import datetime as datetime
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from scipy.stats import multivariate_normal, norm, t

warnings.filterwarnings("ignore")
from math import *

from arch import arch_model
from pypfopt import HRPOpt, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from scipy.optimize import minimize
from sklearn.decomposition import PCA

# from maxt import multskewt

os.chdir("/home/sahil/diss")
k = pd.read_csv("djia.csv")
# Convert string to datetime datatype consisting of only dates and not time
k["Date"] = pd.to_datetime(k.Date).apply(lambda x: x.date())


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


def likelihood_skew_t(y, alpha, beta, omega, skew, v, init_vol):
    # Dimension of data
    d = y.shape[1]

    def minimizer(params):
        v = params[120]
        loglike = []
        for j in range(d):
            vol = [init_vol[j]]
            for i in range(1, len(y)):
                alpha = params[j]
                beta = params[30 + j]
                omega = params[60 + j]
                skew = params[90 + j]
                cond_vol = beta * vol[i - 1] + alpha * y[i - 1][j] ** 2 + omega
                if (alpha + beta > 1) and omega < 0:
                    return 1000
                if cond_vol <= 0:
                    return 1000
                vol.append(cond_vol)
                pdf = np.float(
                    multskewt(
                        y[i],
                        0,
                        cond_vol,
                        skew,
                        v,
                    )
                )

                loglike.append(-np.log(pdf))
        return np.array(loglike).mean()

    res = minimize(
        minimizer,
        alpha + beta + omega + skew + [v],
        method="nelder-mead",
        bounds=((0.01, 1),) * 60
        + ((0.005, 1),) * 30
        + ((-1, 1),) * 30
        + ((2.1, 100)),
        tol=1e-6,
    )
    return res


def gen_ret(df):
    assets = [
        i for i in df.columns if i != "Date"
    ]  # Extracting Column names of all assets

    for asset in assets:
        df[asset + "_logret"] = (
            np.log(df[asset] / df[asset].shift(1)) * 100
        )  # Calculating log returns for an asset
        # df[asset + "_perret"] = (
        #    df[asset].pct_change() * 100
        # )  # Calculating percentage Returns for an asset
    # perret_cols = [i for i in df.columns if "_perret" in i]
    return df


def cov_est(pvar, w, lamb):
    # var = np.diag(list(pvar[0:3]))
    # # var = np.diag(pvar[0:3])
    # w_trans = w[3:]
    # w_left = w[:][3:]

    return w.T * np.diag(pvar) * w


def predict(y, omega, alpha, beta):
    y = np.array(y)
    vol = [0]
    for i in range(1, len(y)):
        cond_vol = beta * vol[i - 1] + alpha * y[i - 1] ** 2 + omega
        vol.append(cond_vol)
    return vol


k = gen_ret(k)
k = k.dropna()
lnret = [i for i in k.columns if "_logret" in i]

pca = PCA(n_components=len(lnret))
krets = k[lnret + ["Date"]]
ktrain = krets[krets.Date < datetime.date(2019, 6, 1)]
ktest = krets[krets.Date > datetime.date(2019, 6, 1)]
# Store lenght of training set
tlen = len(ktrain)
# Column names of principal components
pcols = ["p" + str(i + 1) for i in range(29)]
# Column names of Garch volatility of principal components
pcolsvar = [f"p{i+1}_vol" for i in range(29)]
# Extract the principal components from train set and
# do the transformation on the entire set using eigenvectors
# obtained from the training set
pca = pca.fit(ktrain[lnret])
k[pcols] = pca.transform(k[lnret])
ktrain[pcols] = k[pcols].iloc[0:tlen]
ktest[pcols] = k[pcols].iloc[tlen:]
# Extract the eigenvectors
w = np.matrix(pca.components_)
# Get the eigenvalues
lamb = pca.explained_variance_
loglike = 0
for i in range(29):
    model = arch_model(
        ktrain[pcols[i]],
        mean="Zero",
        vol="GARCH",
        p=1,
        o=0,
        q=1,
    )
    mod = model.fit()
    loglike = loglike + mod.loglikelihood
    ktrain[pcolsvar[i]] = mod.conditional_volatility ** 2
    alpha = mod.params["alpha[1]"]
    beta = mod.params["beta[1]"]
    omega = mod.params["omega"]
    preds = predict(k[pcols[i]], omega, alpha, beta)
    k[pcolsvar[i]] = preds
    ktrain[f"{pcolsvar[i]}_nonlib"] = preds[:tlen]
    ktest[pcolsvar[i]] = preds[tlen:]
print(f"Normal Loglikelihood={loglike/(len(ktrain))}")
# Extracting time varying covariance matrix

k["cov_t"] = k[pcolsvar].apply(lambda row: cov_est(row, w, lamb), axis=1)

# Save for further processing
k.to_pickle("returns_pca.pkl")

ktrain["cov_t"] = ktrain[pcolsvar].apply(
    lambda row: cov_est(row, w, lamb), axis=1
)

ktest["cov_t"] = ktest[pcolsvar].apply(
    lambda row: cov_est(row, w, lamb), axis=1
)

ktrain["log_t"] = ktrain.apply(
    lambda row: multivariate_normal.logpdf(
        row[lnret], [0] * len(pcols), row["cov_t"]
    ),
    axis=1,
)
ktest["log_t"] = ktest.apply(
    lambda row: multivariate_normal.logpdf(
        row[lnret], [0] * len(pcols), row["cov_t"]
    ),
    axis=1,
)

v = 8.9
ktrain["log_t_tdist"] = ktrain.apply(
    lambda row: np.log(
        multivariate_t_distribution(
            row[lnret],
            [0] * len(pcols),
            row["cov_t"],
            v,
            len(pcols),
        )
    ),
    axis=1,
)
ktest["log_t_tdist"] = ktest.apply(
    lambda row: np.log(
        multivariate_t_distribution(
            row[lnret],
            [0] * len(pcols),
            row["cov_t"],
            v,
            len(pcols),
        )
    ),
    axis=1,
)

print(ktrain.log_t.mean())
print(ktrain.log_t_tdist.mean())
