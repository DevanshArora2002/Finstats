import numpy as np
import pandas as pd
from scipy import stats


# OLS of y on X. X already contains a column of ones when an intercept is wanted.
def ols(y, X, names=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    T, k = X.shape
    if names is None:
        names = [f"x{j + 1}" for j in range(k)]

    # (X'X)^{-1} and (X'X)^{-1} X'y
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    RSS = float(np.sum(resid**2))
    s2 = RSS / (T - k)
    V = s2 * XtX_inv
    se = np.sqrt(np.diag(V))
    tval = beta / se
    df = T - k
    pval = 2 * stats.t.sf(np.abs(tval), df)
    TSS = float(np.sum((y - np.mean(y)) ** 2))
    R2 = 1 - RSS / TSS
    R2adj = 1 - (T - 1) / (T - k) * (1 - R2)

    index = list(names)
    return {
        "beta": pd.Series(beta, index=index),
        "se": pd.Series(se, index=index),
        "t": pd.Series(tval, index=index),
        "p": pd.Series(pval, index=index),
        "V": pd.DataFrame(V, index=index, columns=index),
        "XtX_inv": XtX_inv,
        "resid": resid,
        "fitted": y - resid,
        "RSS": RSS,
        "s2": s2,
        "T": int(T),
        "k": int(k),
        "R2": float(R2),
        "R2adj": float(R2adj),
        "X": X,
        "y": y,
    }


def _restriction_matrix(R, k):
    R = np.asarray(R, dtype=float)
    if R.ndim == 1:
        return R.reshape((-1, k), order="F")
    if R.shape[1] != k:
        return R.reshape((-1, k), order="F")
    return R


# F test of H0: R beta = r (q linear restrictions).
# Classical version (V is None): exact F(q, T-k) under Gauss-Markov and normal errors.
# With a robust V it returns the Wald statistic divided by q.
def f_test(fit, R, r=None, V=None):
    R = _restriction_matrix(R, fit["k"])
    q = R.shape[0]
    if r is None:
        r = np.zeros(q)
    r = np.asarray(r, dtype=float).reshape(q)

    d = R @ fit["beta"].to_numpy() - r
    if V is None:
        middle = R @ fit["XtX_inv"] @ R.T
        scale = q * fit["s2"]
    else:
        middle = R @ np.asarray(V, dtype=float) @ R.T
        scale = q
    Fstat = float(d @ np.linalg.solve(middle, d) / scale)
    df2 = fit["T"] - fit["k"]
    return {
        "F": Fstat,
        "df1": int(q),
        "df2": int(df2),
        "d": float(d[0]) if q == 1 else d,
        "p": float(stats.f.sf(Fstat, q, df2)),
        "crit": float(stats.f.ppf(0.95, q, df2)),
    }


# t test of H0: beta_j = b0 (two-sided). se can be replaced by robust standard errors.
def t_test(fit, j, b0=0.0, se=None):
    if se is None:
        se = fit["se"]
    if not isinstance(se, pd.Series):
        se = pd.Series(np.asarray(se, dtype=float).reshape(-1), index=fit["beta"].index)
    df = fit["T"] - fit["k"]
    tval = float((fit["beta"][j] - b0) / se[j])
    return {
        "t": tval,
        "df": int(df),
        "p": float(2 * stats.t.sf(abs(tval), df)),
        "crit": float(stats.t.ppf(0.975, df)),
    }


# White heteroskedasticity-robust covariance matrix (HC1).
def vcov_white(fit):
    meat = (fit["X"] * fit["resid"][:, None]).T @ (fit["X"] * fit["resid"][:, None])
    scale = fit["T"] / (fit["T"] - fit["k"])
    V = scale * fit["XtX_inv"] @ meat @ fit["XtX_inv"]
    names = list(fit["beta"].index)
    return pd.DataFrame(V, index=names, columns=names)
