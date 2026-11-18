import numpy as np
import pandas as pd
from scipy import stats

from ols_functions import ols


# Hit ratio Z(alpha): share of months t = 2..n whose sign is called right.
# phat estimates P(R_t > 0 | X_{t-1}); R is the realised return.
def hit_ratio(phat, R, alpha=0.5):
    phat = np.asarray(phat, dtype=float)
    R = np.asarray(R, dtype=float)
    called = (phat > alpha) & (R > 0) | (phat <= alpha) & (R <= 0)
    return float(np.mean(called))


# Fit Y_{t+1} = beta' X_t + eps_t by OLS for one set of predictors.
# P is the n x K matrix of candidate predictors, cols the chosen columns.
# Returns the OLS fit and the fitted probabilities for t = 2..n.
def lpm_fit(Y, P, cols):
    P = pd.DataFrame(P)
    Y = np.asarray(Y, dtype=float)
    n = len(Y)
    cols = np.asarray(cols, dtype=int).reshape(-1)
    blocks = [np.ones(n - 1)]
    names = ["const"]
    if cols.size:
        blocks.append(P.iloc[: n - 1, cols].to_numpy(dtype=float))
        names.extend(P.columns[j] for j in cols)
    X = np.column_stack(blocks)
    fit = ols(Y[1:], X, names=names)
    return {"fit": fit, "phat": fit["fitted"]}


# All 2^K combinations of predictors. The constant is always included.
# Records Z(alpha), the number of predictors, and adjusted R^2.
# Best means highest Z, then fewer predictors, then higher adjusted R^2.
def search_subsets(Y, R, P, alpha=0.5):
    P = pd.DataFrame(P)
    Y = np.asarray(Y, dtype=float)
    R = np.asarray(R, dtype=float)
    K = P.shape[1]
    n = len(Y)
    rows = []
    for mask in range(2**K):
        cols = np.flatnonzero((mask & (1 << np.arange(K))) > 0)
        fit = lpm_fit(Y, P, cols)
        label = ", ".join(P.columns[j] for j in cols)
        rows.append(
            {
                "mask": mask,
                "k": int(cols.size),
                "Z": hit_ratio(fit["phat"], R[1:], alpha),
                "R2adj": fit["fit"]["R2adj"],
                "vars": label if label else "(constant only)",
            }
        )
    res = pd.DataFrame(rows)
    return res.sort_values(
        ["Z", "k", "R2adj"],
        ascending=[False, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


# Exact maximisation of Z(alpha) over 0 < alpha < 1.
# Z(alpha) changes only when alpha crosses a fitted value, so it is constant
# on [p_(j), p_(j+1)). Evaluating one point in each interval gives the maximum.
def alpha_scan(phat, R):
    phat = np.asarray(phat, dtype=float)
    interior = np.sort(np.unique(phat[(phat > 0) & (phat < 1)]))
    lo = np.concatenate(([0.0], interior))
    hi = np.concatenate((interior, [1.0]))
    at = np.concatenate(([interior[0] / 2], interior))
    Z = np.array([hit_ratio(phat, R, a) for a in at])
    best = np.flatnonzero(Z == Z.max())
    return {
        "Zmax": float(Z.max()),
        "intervals": np.column_stack((lo[best], hi[best])),
        "table": pd.DataFrame({"lo": lo, "hi": hi, "Z": Z}),
    }


# Pesaran-Timmermann (1992) test of directional accuracy.
# H0: the predicted signs are independent of the realised signs.
def pt_test(phat, R, alpha=0.5):
    phat = np.asarray(phat, dtype=float)
    R = np.asarray(R, dtype=float)
    n = len(R)
    up = R > 0
    pu = phat > alpha
    P_hat = float(np.mean(up == pu))
    Py = float(np.mean(up))
    Px = float(np.mean(pu))
    Ps = Py * Px + (1 - Py) * (1 - Px)
    VP = Ps * (1 - Ps) / n
    VPs = (
        (2 * Py - 1) ** 2 * Px * (1 - Px) + (2 * Px - 1) ** 2 * Py * (1 - Py)
    ) / n + 4 * Py * Px * (1 - Py) * (1 - Px) / n**2
    stat = (P_hat - Ps) / np.sqrt(VP - VPs)
    return pd.Series({"stat": float(stat), "p": float(stats.norm.sf(stat))})


def _best_columns(ytr, Ptr, Rtr, alpha):
    # Same scores as search_subsets, from the normal equations on one window.
    T, K = Ptr.shape
    design = np.column_stack((np.ones(T), Ptr))
    gram = design.T @ design
    gy = design.T @ ytr
    tss = float(np.sum((ytr - np.mean(ytr)) ** 2))
    best_key = None
    best_cols = np.array([], dtype=int)
    powers = 1 << np.arange(K)
    for mask in range(2**K):
        cols = np.flatnonzero((mask & powers) > 0)
        idx = np.concatenate(([0], cols + 1))
        beta = np.linalg.solve(gram[np.ix_(idx, idx)], gy[idx])
        fitted = design[:, idx] @ beta
        residual = ytr - fitted
        k = idx.size
        score = hit_ratio(fitted, Rtr, alpha)
        R2adj = 1 - (T - 1) / (T - k) * float(np.sum(residual**2)) / tss
        key = (-score, k, -R2adj, mask)
        if best_key is None or key < best_key:
            best_key = key
            best_cols = cols
    return best_cols


# Expanding-window hit ratio at alpha = 1/2.
# At each date t >= t0 the model uses only pairs (Y_{s+1}, X_s) with s < t,
# and the probability for t+1 is x_t' beta-hat.
# cols = a fixed set of predictors. cols = None re-selects the best set
# in every window, with the same rule as search_subsets.
def oos_hit(Y, R, P, t0, cols=None, alpha=0.5):
    P = pd.DataFrame(P)
    names = list(P.columns)
    P = P.to_numpy(dtype=float)
    Y = np.asarray(Y, dtype=float)
    R = np.asarray(R, dtype=float)
    n = len(Y)
    if cols is not None:
        cols = np.asarray(cols, dtype=int).reshape(-1)

    calls = []
    chosen = []
    phat = []
    for t in range(t0, n - 1):
        ytr = Y[1 : t + 1]
        Ptr = P[:t, :]
        if cols is None:
            pick = _best_columns(ytr, Ptr, R[1 : t + 1], alpha)
        else:
            pick = cols
        if pick.size:
            X = np.column_stack((np.ones(len(ytr)), Ptr[:, pick]))
            x_next = np.concatenate(([1.0], P[t, pick]))
        else:
            X = np.ones((len(ytr), 1))
            x_next = np.ones(1)
        beta = np.linalg.solve(X.T @ X, X.T @ ytr)
        p_next = float(x_next @ beta)
        realised = R[t + 1]
        hit = (p_next > alpha and realised > 0) or (p_next <= alpha and realised <= 0)
        calls.append(hit)
        phat.append(p_next)
        chosen.append(", ".join(names[j] for j in pick))

    return {
        "Z": float(np.mean(calls)),
        "n": len(calls),
        "chosen": chosen,
        "phat": np.asarray(phat, dtype=float),
        "R": R[t0 + 1 : n],
    }
