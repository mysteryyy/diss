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
from sklearn.neighbors import KernelDensity

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


def mpPDF(var, q, pts):
    # Marcenko-Pastur pdf
    # q=T/N
    eMin, eMax = (
        var * (1 - (1.0 / q) ** 0.5) ** 2,
        var * (1 + (1.0 / q) ** 0.5) ** 2,
    )
    eVal = np.linspace(eMin, eMax, pts)
    pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** 0.5
    pdf = pd.Series(pdf, index=eVal)
    return pdf


def getPCA(matrix):
    # Get eVal,eVec from a Hermitian matrix
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[indices], eVec[:, indices]
    eVal = np.diagflat(eVal)
    return eVal, eVec


def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
    return corr


def fitKDE(obs, bWidth=0.25, kernel="gaussian", x=None):
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
    if x is None:
        x = np.unique(obs).reshape(-1, 1)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    logProb = kde.score_samples(x)  # log(density)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())
    return pdf


def errPDFs(var, eVal, q, bWidth, pts=1000):
    # Fit error
    var = var[0]
    pdf0 = mpPDF(var, q, pts)  # theoretical pdf
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values)  # empirical pdf
    sse = np.sum((pdf1 - pdf0) ** 2)
    return sse


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def findMaxEval(eVal, q, bWidth):
    # Find max random eVal by fitting Marcenko’s dist
    out = minimize(
        lambda *x: errPDFs(*x),
        0.5,
        args=(eVal, q, bWidth),
        bounds=((1e-5, 1 - 1e-5),),
    )
    if out["success"]:
        var = out["x"][0]
    else:
        var = 1
    eMax = var * (1 + (1.0 / q) ** 0.5) ** 2
    return eMax, var
    # - - - - - - - - - - - - - - - - - - - -


def denoisedCorr(eVal, eVec, nFacts):
    # Remove noise from corr by fixing random eigenvalues
    eVal_ = np.diag(eVal).copy()
    eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0] - nFacts)
    eVal_ = np.diag(eVal_)
    corr1 = np.dot(eVec, eVal_).dot(eVec.T)
    corr1 = cov2corr(corr1)
    return corr1
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# corr1=denoisedCorr(eVal0,eVec0,nFacts0)
# eVal1,eVec1=getPCA(corr1)
# ret_col = [i for i in k.columns if "_logret" in i]
# cov1 = k[ret_col].corr()
# eVal0, evec = getPCA(cov1)
# x = np.matrix(k[ret_col])
# q = x.shape[0] / float(x.shape[1])
# eMax0, var0 = findMaxEval(np.diag(eVal0), q=q, bWidth=0.01)
# nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
# print(nFacts0)
