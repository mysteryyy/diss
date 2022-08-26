---
title: "\\vspace{-5.6cm}**Dissertation Methodology Overview**"
author: "\\vspace{-5.5cm}**Sahil Singh**"
header-includes:
   - \usepackage{fancyvrb}
   - \usepackage{amsmath,amssymb,amsthm}
   - \usepackage{environ}
   - \usepackage{verbatim}
   - \usepackage[backend=biber]{biblatex}
   - \usepackage{booktabs}
   - \addbibresource{.bib}
output:
    pdf_document
---

# Overview

The purpose of this work is to model time varying distribution of portoflio of equities using fat tailed distribution. An attempt has been made to model stylized facts of equity returns like time varying volatility and correlation through the use of Orthogonal Garch model and modifications have been made to ensure that it is compatible with the fat-tailed distributions being used. In addition, alternate and new methodologies have been explored such as Covariance Shrinkage using RMT or Ledoit-Wolf methods and linear regression using principal components.

# Methodology 

In the literature, multiple fat tailed distributions have been explored to fit the multivariate returns and some have found the Student T and Assymetric Student T to be the best fit for this kind of data. For fitting the data to multivariate skew T distribution,

* Daily log returns of 30 stocks from Dow Jones 30 Index is taken and their principal components are extracted.

* A feature of skewed t distribution is that it is invariant under linear transformation. So given a skew t distribution $X \sim \textrm{SkewT}(\mu,\Sigma,\gamma,\nu)$ where $\mu$ is the location parameter, $\Sigma$ is the dispersion matrix, $\gamma$ is the skewness parameter and $\nu$ is the degree of freedom, under linear transformation of X, $Y=w^{T}X$ it follows that $Y \sim  \text{SkewT}(w^{T}mu,w^{T}\Sigma w,w^{T}\gamma,\nu)$  which is a univariate distribution, Hence it follows that the the principal components will also have a univariate distribution since they are a linear transformation of the returns. 

* Following these results univariate skew T garch is fit on each of the principal components, keeping the degree of freedom parameter fixed(using the same degree of freedom parameter for all principal components ). Each principal component is fit seperately where the $\nu$ is kept constant and the log likelihood from all the compinents is averaged. The $\nu$ value is then changed and the process is repeated. In this way, we obtain different log likelihood values for different $\nu$ and the one which maximises the log-likelihood is choosen as the paramater for the multvariate distribution.

* Let $w$ be the eigenvector found from the spectral decomposition.Let $\lambda_{t}$ represent the $k \times k$ diagonal matrix such that the $k^{th}$ diagonal entries have the Garch predicted conditional volatility of $k^{th}$ principal component at time t. The $\Sigma_{t}$ is calculated as $w^{T} \lambda w$ owing to the relation between $Y$ and $X$ given abpve.Let the $\gamma^{'}$ be the vector of skewness of the principal components. Then, the time invariant skewness vector for the multivariate distribution is given as $w\gamma^{'}$(Since principal component skewness is obtained from the $Ax=B$ transformation where $A$ is $w^{T}$, $x$ is $\gamma$ and $B$ is $\gamma^{'}$, therefore $x=A^{-1}B$ and since $w^{-1}=w^{T}$, we get the equation). 

* $\nu^{`}$ 
