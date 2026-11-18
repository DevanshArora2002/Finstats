import calendar
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.display import display
from scipy import stats

from src.ols_functions import f_test, ols, t_test, vcov_white
from src.table_helpers import clear_macros, fmt, fmtp, set_macro, write_macros


def simple_ret(p):
    p = np.asarray(p, dtype=float)
    out = np.empty(len(p))
    out[0] = np.nan
    out[1:] = p[1:] / p[:-1] - 1
    return out


def jarque_bera(e):
    e = np.asarray(e, dtype=float)
    n = len(e)
    z = e - np.mean(e)
    m2 = np.mean(z**2)
    S = np.mean(z**3) / m2**1.5
    K = np.mean(z**4) / m2**2
    JB = n / 6 * (S**2 + (K - 3) ** 2 / 4)
    return pd.Series(
        {"S": S, "K": K, "JB": JB, "p": float(stats.chi2.sf(JB, 2))}
    )


def coef_frame(fit):
    return pd.DataFrame(
        {"est": fit["beta"], "se": fit["se"], "t": fit["t"], "p": fit["p"]}
    )


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    clear_macros()

    # Import the workbook. Text in the price column is the note under the header.
    raw = pd.read_excel("data/data_coursework1_Q1__1_.xls", header=0)
    dat = pd.DataFrame(
        {
            "year": pd.to_numeric(raw["year_"], errors="coerce"),
            "month": pd.to_numeric(raw["month_"], errors="coerce"),
            "tbill": pd.to_numeric(raw["1-month Tbill"], errors="coerce"),
            "sp500": pd.to_numeric(raw["SP500"], errors="coerce"),
            "ibm": pd.to_numeric(raw["IBM"], errors="coerce"),
        }
    )
    dat = dat.loc[dat["year"].notna()].reset_index(drop=True)

    dat["rM"] = simple_ret(dat["sp500"])
    dat["ri"] = simple_ret(dat["ibm"])
    dat["rf"] = dat["tbill"] / 100

    # Feb 1962 - Dec 1990: the first IBM return needs the January 1962 price.
    s = dat.loc[dat["ri"].notna() & dat["rM"].notna()].reset_index(drop=True)
    month_index = s["year"] * 12 + s["month"]
    if not np.all(np.diff(month_index.to_numpy()) == 1):
        raise AssertionError("Sample months are not consecutive")

    y = (s["ri"] - s["rf"]).to_numpy(dtype=float)
    x = (s["rM"] - s["rf"]).to_numpy(dtype=float)
    D = (x > 0).astype(float)
    T = len(y)

    display(s[["year", "month", "ri", "rM", "rf"]].head())

    # Model (1): CAPM
    X1 = np.column_stack((np.ones(T), x))
    m1 = ols(y, X1, names=["alpha", "beta"])

    # Model (2): up/down betas and a quadratic term
    X2 = np.column_stack((np.ones(T), D * x, (1 - D) * x, x**2))
    m2 = ols(y, X2, names=["alpha", "beta1", "beta2", "beta3"])

    beta_lm1 = np.asarray(sm.OLS(y, X1).fit().params, dtype=float)
    beta_lm2 = np.asarray(sm.OLS(y, X2).fit().params, dtype=float)
    if not np.allclose(m1["beta"].to_numpy(), beta_lm1, rtol=1e-8, atol=1e-8):
        raise AssertionError("Model (1) does not match statsmodels OLS")
    if not np.allclose(m2["beta"].to_numpy(), beta_lm2, rtol=1e-8, atol=1e-8):
        raise AssertionError("Model (2) does not match statsmodels OLS")

    # F test of H0: beta1 = beta2 in model (2)
    R = np.array([[0.0, 1.0, -1.0, 0.0]])
    F_wald = f_test(m2, R)
    X2r = np.column_stack((np.ones(T), x, x**2))
    m2r = ols(y, X2r, names=["alpha", "beta", "beta3"])
    F_rss = ((m2r["RSS"] - m2["RSS"]) / 1) / (m2["RSS"] / (T - 4))
    if abs(F_wald["F"] - F_rss) >= 1e-8:
        raise AssertionError("Wald and RSS forms of the F test disagree")
    F_rob = f_test(m2, R, V=vcov_white(m2))
    se_d = float(np.sqrt(m2["s2"] * (R @ m2["XtX_inv"] @ R.T).item()))
    t_b12 = F_wald["d"] / se_d

    # t tests of H0: alpha = 0
    ta1 = t_test(m1, "alpha")
    ta2 = t_test(m2, "alpha")
    ta1r = t_test(m1, "alpha", se=np.sqrt(np.diag(vcov_white(m1))))
    ta2r = t_test(m2, "alpha", se=np.sqrt(np.diag(vcov_white(m2))))

    jb1 = jarque_bera(m1["resid"])
    jb2 = jarque_bera(m2["resid"])

    vif = []
    for j in range(1, 4):
        others = [col for col in range(4) if col != j]
        aux = ols(X2[:, j], X2[:, others])
        vif.append(1 / (1 - aux["R2"]))
    vif = np.asarray(vif, dtype=float)
    cor_down_quad = float(np.corrcoef(X2[:, 2], X2[:, 3])[0, 1])

    # Model (2) without the quadratic term
    X2b = np.column_stack((np.ones(T), D * x, (1 - D) * x))
    m2b = ols(y, X2b, names=["alpha", "beta1", "beta2"])
    F_2b = f_test(m2b, [0, 1, -1])

    # Cook's distance, and model (2) without the most influential month
    h = np.sum((X2 @ m2["XtX_inv"]) * X2, axis=1)
    cook = m2["resid"] ** 2 * h / (m2["k"] * m2["s2"] * (1 - h) ** 2)
    iout = int(np.argmax(cook))
    m2o = ols(np.delete(y, iout), np.delete(X2, iout, axis=0), names=list(m2["beta"].index))
    F_2o = f_test(m2o, R)

    # Oct and Nov 1987
    crash = np.flatnonzero((s["year"].to_numpy() == 1987) & np.isin(s["month"].to_numpy(), [10, 11]))
    m2c = ols(np.delete(y, crash), np.delete(X2, crash, axis=0), names=list(m2["beta"].index))
    F_2c = f_test(m2c, R)

    # Residual autocorrelation of model (1)
    e = m1["resid"] - np.mean(m1["resid"])
    ac = np.array(
        [np.sum(e[j:] * e[: T - j]) / np.sum(e**2) for j in range(1, 7)]
    )
    LB = T * (T + 2) * np.sum(ac**2 / (T - np.arange(1, 7)))
    LBp = float(stats.chi2.sf(LB, 6))

    # IBM is a month-end price; the S&P 500 column is a monthly average.
    lead = ols(
        y[:-1],
        np.column_stack((np.ones(T - 1), x[:-1], x[1:])),
        names=["alpha", "x", "x_lead"],
    )
    cor_lead = float(np.corrcoef(s["ri"].to_numpy()[:-1], s["rM"].to_numpy()[1:])[0, 1])
    cor_lag = float(np.corrcoef(s["ri"].to_numpy()[1:], s["rM"].to_numpy()[:-1])[0, 1])
    cor_mkt = float(np.corrcoef(s["rM"].to_numpy()[1:], s["rM"].to_numpy()[:-1])[0, 1])

    tb1 = t_test(m1, "beta", b0=1)

    def ci(fit, j):
        width = stats.t.ppf(0.975, fit["T"] - fit["k"]) * fit["se"].iloc[j]
        return fit["beta"].iloc[j] + np.array([-1.0, 1.0]) * width

    def slope_at(g):
        level = m2["beta"].iloc[1] if g > 0 else m2["beta"].iloc[2]
        return float(level + 2 * m2["beta"].iloc[3] * g)

    year = s["year"].to_numpy()
    month = s["month"].to_numpy()
    print(
        f"Sample: {int(year[0])}-{int(month[0]):02d} to "
        f"{int(year[T - 1])}-{int(month[T - 1]):02d}, T = {T}, up months = {int(D.sum())}"
    )
    print("\nModel (1)")
    display(coef_frame(m1).round(5))
    print("\nModel (2)")
    display(coef_frame(m2).round(5))
    print(
        f"F(beta1=beta2) = {F_wald['F']:.4f}, p = {F_wald['p']:.4f} "
        f"(robust {F_rob['F']:.4f}, p = {F_rob['p']:.4f})"
    )
    print(
        f"t(alpha=0): model 1 {ta1['t']:.4f} (p {ta1['p']:.4f}), "
        f"model 2 {ta2['t']:.4f} (p {ta2['p']:.4f})"
    )
    print(
        f"JB: model 1 {jb1['JB']:.1f} (p {jb1['p']:.2g}), "
        f"model 2 {jb2['JB']:.1f} (p {jb2['p']:.2g}); "
        f"VIF {', '.join(f'{v:.2f}' for v in vif)}"
    )
    print(
        f"No quadratic: b1 {m2b['beta'].iloc[1]:.3f} b2 {m2b['beta'].iloc[2]:.3f} "
        f"F {F_2b['F']:.3f} p {F_2b['p']:.4f} | "
        f"without {int(year[iout])}-{int(month[iout]):02d}: "
        f"F {F_2o['F']:.3f} p {F_2o['p']:.4f}"
    )
    print(
        f"Without Oct+Nov 1987: F {F_2c['F']:.3f} p {F_2c['p']:.4f} | "
        f"TM curvature {m2r['beta'].iloc[2]:.3f} (t {m2r['t'].iloc[2]:.2f}) | "
        f"lead {lead['beta'].iloc[2]:.3f} (t {lead['t'].iloc[2]:.2f})"
    )

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    def coef_rows(fit, labels, rob):
        rse = np.sqrt(np.diag(np.asarray(rob, dtype=float)))
        beta = fit["beta"].to_numpy()
        return [
            " & ".join(
                [
                    labels[i],
                    fmt(beta[i]),
                    fmt(fit["se"].iloc[i]),
                    fmt(fit["t"].iloc[i], 3),
                    fmtp(fit["p"].iloc[i]),
                    fmt(beta[i] / rse[i], 3),
                ]
            )
            + " \\\\"
            for i in range(len(labels))
        ]

    def fit_note(fit):
        return (
            "\\multicolumn{6}{l}{\\footnotesize "
            f"$R^2={fmt(fit['R2'], 4)}$, $\\bar R^2={fmt(fit['R2adj'], 4)}$, "
            f"$s={fmt(np.sqrt(fit['s2']), 4)}$, $T={T}$}} \\\\"
        )

    tab = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Parameter & Estimate & Std.\\ error & $t$-stat & $p$-value & robust $t$ \\\\",
        "\\midrule",
        "\\multicolumn{6}{l}{\\textit{Model (1): CAPM}} \\\\",
        *coef_rows(m1, ["$\\alpha$", "$\\beta$"], vcov_white(m1)),
        fit_note(m1),
        "\\midrule",
        "\\multicolumn{6}{l}{\\textit{Model (2): up/down betas and quadratic term}} \\\\",
        *coef_rows(
            m2,
            ["$\\alpha$", "$\\beta_1$ (up)", "$\\beta_2$ (down)", "$\\beta_3$ (quadratic)"],
            vcov_white(m2),
        ),
        fit_note(m2),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    write_lines("results/q1_estimates.tex", tab)

    def dec(p):
        return "Reject" if p < 0.05 else "Do not reject"

    test_tab = [
        "\\begin{tabular}{llrrrl}",
        "\\toprule",
        "Null hypothesis & Version & Statistic & 5\\% crit.\\ value & $p$-value & Decision \\\\",
        "\\midrule",
        (
            f"$\\beta_1=\\beta_2$ (model 2) & classical $F({F_wald['df1']},{F_wald['df2']})$ & "
            f"{fmt(F_wald['F'], 3)} & {fmt(F_wald['crit'], 3)} & {fmtp(F_wald['p'])} & {dec(F_wald['p'])} \\\\"
        ),
        (
            f" & White-robust $F$ & {fmt(F_rob['F'], 3)} & {fmt(F_rob['crit'], 3)} & "
            f"{fmtp(F_rob['p'])} & {dec(F_rob['p'])} \\\\"
        ),
        "\\midrule",
        (
            f"$\\alpha=0$ (model 1) & classical $t({ta1['df']})$ & {fmt(ta1['t'], 3)} & "
            f"$\\pm${fmt(ta1['crit'], 3)} & {fmtp(ta1['p'])} & {dec(ta1['p'])} \\\\"
        ),
        (
            f" & White-robust $t$ & {fmt(ta1r['t'], 3)} & $\\pm${fmt(ta1r['crit'], 3)} & "
            f"{fmtp(ta1r['p'])} & {dec(ta1r['p'])} \\\\"
        ),
        (
            f"$\\alpha=0$ (model 2) & classical $t({ta2['df']})$ & {fmt(ta2['t'], 3)} & "
            f"$\\pm${fmt(ta2['crit'], 3)} & {fmtp(ta2['p'])} & {dec(ta2['p'])} \\\\"
        ),
        (
            f" & White-robust $t$ & {fmt(ta2r['t'], 3)} & $\\pm${fmt(ta2r['crit'], 3)} & "
            f"{fmtp(ta2r['p'])} & {dec(ta2r['p'])} \\\\"
        ),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    write_lines("results/q1_tests.tex", test_tab)

    set_macro("qoneT", T)
    set_macro("qoneUp", int(D.sum()))
    set_macro("qoneDown", T - int(D.sum()))
    set_macro("qoneMeanY", fmt(100 * np.mean(y), 3))
    set_macro("qoneMeanX", fmt(100 * np.mean(x), 3))
    set_macro("qoneSdY", fmt(100 * np.std(y, ddof=1), 3))
    set_macro("qoneSdX", fmt(100 * np.std(x, ddof=1), 3))
    set_macro("qoneAlpha", fmt(m1["beta"].iloc[0], 4))
    set_macro("qoneAlphaPct", fmt(100 * m1["beta"].iloc[0], 3))
    set_macro("qoneAlphaAnn", fmt(1200 * m1["beta"].iloc[0], 2))
    set_macro("qoneBeta", fmt(m1["beta"].iloc[1], 4))
    set_macro("qoneBetaSe", fmt(m1["se"].iloc[1], 4))
    set_macro("qoneRsq", fmt(m1["R2"], 3))
    set_macro("qoneAlphaTwo", fmt(m2["beta"].iloc[0], 4))
    set_macro("qoneAlphaTwoPct", fmt(100 * m2["beta"].iloc[0], 3))
    set_macro("qoneBetaUp", fmt(m2["beta"].iloc[1], 4))
    set_macro("qoneBetaDown", fmt(m2["beta"].iloc[2], 4))
    set_macro("qoneBetaQuad", fmt(m2["beta"].iloc[3], 4))
    set_macro("qoneD", fmt(F_wald["d"], 4))
    set_macro("qoneSeD", fmt(se_d, 4))
    set_macro("qoneF", fmt(F_wald["F"], 3))
    set_macro("qoneFp", fmtp(F_wald["p"]))
    set_macro("qoneFcrit", fmt(F_wald["crit"], 3))
    set_macro("qoneFdf", F_wald["df2"])
    set_macro("qoneTbb", fmt(t_b12, 3))
    set_macro("qoneFrob", fmt(F_rob["F"], 3))
    set_macro("qoneFrobp", fmtp(F_rob["p"]))
    set_macro("qoneRSSu", fmt(m2["RSS"], 5))
    set_macro("qoneRSSr", fmt(m2r["RSS"], 5))
    set_macro("qoneTa", fmt(ta1["t"], 3))
    set_macro("qoneTap", fmtp(ta1["p"]))
    set_macro("qoneTcrit", fmt(ta1["crit"], 3))
    set_macro("qoneTdf", ta1["df"])
    set_macro("qoneTaTwo", fmt(ta2["t"], 3))
    set_macro("qoneTaTwop", fmtp(ta2["p"]))
    set_macro("qoneTcritTwo", fmt(ta2["crit"], 3))
    set_macro("qoneTdfTwo", ta2["df"])
    set_macro("qoneTaRob", fmt(ta1r["t"], 3))
    set_macro("qoneTaRobp", fmtp(ta1r["p"]))
    set_macro("qoneTaTwoRob", fmt(ta2r["t"], 3))
    set_macro("qoneTaTwoRobp", fmtp(ta2r["p"]))

    out_month = int(month[iout])
    out_year = int(year[iout])
    rob_tab = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Specification & $\\hat\\beta_1$ & $\\hat\\beta_2$ & $\\hat\\beta_3$ & $F$ ($\\beta_1=\\beta_2$) & $p$-value \\\\",
        "\\midrule",
        (
            f"Model (2), full sample & {fmt(m2['beta'].iloc[1], 3)} & {fmt(m2['beta'].iloc[2], 3)} & "
            f"{fmt(m2['beta'].iloc[3], 3)} & {fmt(F_wald['F'], 3)} & {fmtp(F_wald['p'])} \\\\"
        ),
        (
            f"Model (2) without $\\beta_3$ & {fmt(m2b['beta'].iloc[1], 3)} & {fmt(m2b['beta'].iloc[2], 3)} & "
            f"-- & {fmt(F_2b['F'], 3)} & {fmtp(F_2b['p'])} \\\\"
        ),
        (
            f"Model (2) without {calendar.month_abbr[out_month]} {out_year} & "
            f"{fmt(m2o['beta'].iloc[1], 3)} & {fmt(m2o['beta'].iloc[2], 3)} & {fmt(m2o['beta'].iloc[3], 3)} & "
            f"{fmt(F_2o['F'], 3)} & {fmtp(F_2o['p'])} \\\\"
        ),
        (
            f"Model (2) without Oct and Nov 1987 & {fmt(m2c['beta'].iloc[1], 3)} & "
            f"{fmt(m2c['beta'].iloc[2], 3)} & {fmt(m2c['beta'].iloc[3], 3)} & "
            f"{fmt(F_2c['F'], 3)} & {fmtp(F_2c['p'])} \\\\"
        ),
        "\\midrule",
        (
            "Restricted, $\\beta_1=\\beta_2$ (Treynor--Mazuy) & "
            f"\\multicolumn{{2}}{{c}}{{{fmt(m2r['beta'].iloc[1], 3)}}} & "
            f"{fmt(m2r['beta'].iloc[2], 3)} & -- & -- \\\\"
        ),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    write_lines("results/q1_robustness.tex", rob_tab)

    set_macro("qoneJBone", fmt(jb1["JB"], 2))
    set_macro("qoneJBonep", fmtp(jb1["p"]))
    set_macro("qoneSkew", fmt(jb1["S"], 2))
    set_macro("qoneKurt", fmt(jb1["K"], 2))
    set_macro("qoneJBtwo", fmt(jb2["JB"], 2))
    set_macro("qoneJBtwop", fmtp(jb2["p"]))
    set_macro("qoneVIFmax", fmt(float(np.max(vif)), 1))
    set_macro("qoneCorDQ", fmt(cor_down_quad, 2))
    set_macro("qoneBetaUpSe", fmt(m2["se"].iloc[1], 3))
    set_macro("qoneBetaDownSe", fmt(m2["se"].iloc[2], 3))
    set_macro("qoneBetaUpNQ", fmt(m2b["beta"].iloc[1], 3))
    set_macro("qoneBetaDownNQ", fmt(m2b["beta"].iloc[2], 3))
    set_macro("qoneFNQ", fmt(F_2b["F"], 3))
    set_macro("qoneFNQp", fmtp(F_2b["p"]))
    set_macro("qoneOutMonth", f"{calendar.month_name[out_month]} {out_year}")
    set_macro("qoneOutCook", fmt(cook[iout], 3))
    set_macro("qoneOutLev", fmt(h[iout], 3))
    set_macro("qoneAvgLev", fmt(4 / T, 3))
    set_macro("qoneFcr", fmt(F_2c["F"], 3))
    set_macro("qoneFcrp", fmtp(F_2c["p"]))
    set_macro("qoneTMcurv", fmt(m2r["beta"].iloc[2], 3))
    set_macro("qoneTMcurvT", fmt(m2r["t"].iloc[2], 2))
    set_macro("qoneACone", fmt(ac[0], 3))
    set_macro("qoneLB", fmt(LB, 2))
    set_macro("qoneLBp", fmtp(LBp))
    set_macro("qoneLeadCoef", fmt(lead["beta"].iloc[2], 3))
    set_macro("qoneLeadT", fmt(lead["t"].iloc[2], 2))
    set_macro("qoneLeadCont", fmt(lead["beta"].iloc[1], 3))
    set_macro("qoneLeadSum", fmt(lead["beta"].iloc[1] + lead["beta"].iloc[2], 2))
    set_macro("qoneCorLead", fmt(cor_lead, 2))
    set_macro("qoneCorLag", fmt(cor_lag, 2))
    set_macro("qoneACmkt", fmt(cor_mkt, 2))
    set_macro("qoneTbOne", fmt(tb1["t"], 2))
    set_macro("qoneTbOnep", fmtp(tb1["p"]))
    set_macro("qoneCIbLo", fmt(ci(m1, 1)[0], 3))
    set_macro("qoneCIbHi", fmt(ci(m1, 1)[1], 3))
    set_macro("qoneCIaLo", fmt(100 * ci(m1, 0)[0], 2))
    set_macro("qoneCIaHi", fmt(100 * ci(m1, 0)[1], 2))
    set_macro("qoneMeanRM", fmt(100 * np.mean(s["rM"]), 3))
    set_macro("qoneMeanRF", fmt(100 * np.mean(s["rf"]), 3))
    set_macro("qoneRbarOne", fmt(m1["R2adj"], 3))
    set_macro("qoneRbarTwo", fmt(m2["R2adj"], 3))
    set_macro("qoneBetaQuadp", fmtp(m2["p"].iloc[3], 3))
    set_macro("qoneAlphaNQPct", fmt(100 * m2b["beta"].iloc[0], 2))
    set_macro("qoneBetaUpSeNQ", fmt(m2b["se"].iloc[1], 3))
    set_macro("qoneBetaDownSeNQ", fmt(m2b["se"].iloc[2], 3))
    set_macro("qoneFout", fmt(F_2o["F"], 3))
    set_macro("qoneFoutp", fmtp(F_2o["p"]))
    oct_ret = float(dat.loc[(dat["year"] == 1987) & (dat["month"] == 10), "rM"].iloc[0])
    nov_ret = float(dat.loc[(dat["year"] == 1987) & (dat["month"] == 11), "rM"].iloc[0])
    set_macro("qoneOctRet", fmt(100 * oct_ret, 1))
    set_macro("qoneNovRet", fmt(100 * nov_ret, 1))
    set_macro("qoneSlopeUpFive", fmt(slope_at(0.05), 2))
    set_macro("qoneSlopeUpTen", fmt(slope_at(0.10), 2))
    set_macro("qoneSlopeDnFive", fmt(slope_at(-0.05), 2))
    set_macro("qoneSlopeDnTen", fmt(slope_at(-0.10), 2))
    write_macros("results/q1_macros.tex")

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.scatter(100 * x, 100 * y, s=12, c="#8C8C8C", linewidths=0)
    ax.axhline(0, color="#CCCCCC", ls=":")
    ax.axvline(0, color="#CCCCCC", ls=":")
    g = np.linspace(x.min(), x.max(), 400)
    ax.plot(100 * g, 100 * (m1["beta"].iloc[0] + m1["beta"].iloc[1] * g), lw=2, c="black", label="Model (1): CAPM")
    f2 = (
        m2["beta"].iloc[0]
        + m2["beta"].iloc[1] * g * (g > 0)
        + m2["beta"].iloc[2] * g * (g <= 0)
        + m2["beta"].iloc[3] * g**2
    )
    ax.plot(100 * g, 100 * f2, lw=2, ls="--", c="#3A5FCD", label="Model (2): up/down betas + quadratic")
    ax.set_xlabel("Market excess return (%, monthly)")
    ax.set_ylabel("IBM excess return (%, monthly)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig("figures/q1_scatter.pdf")
    plt.close(fig)

    return {
        "m1": m1,
        "m2": m2,
        "m2r": m2r,
        "m2b": m2b,
        "F_wald": F_wald,
        "F_rob": F_rob,
        "ta1": ta1,
        "ta2": ta2,
        "ta1r": ta1r,
        "ta2r": ta2r,
        "jb1": jb1,
        "jb2": jb2,
    }


if __name__ == "__main__":
    main()
