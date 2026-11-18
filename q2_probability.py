import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from src.ols_functions import vcov_white
from src.q2_functions import alpha_scan, hit_ratio, lpm_fit, oos_hit, pt_test, search_subsets
from src.table_helpers import clear_macros, fmt, fmtp, set_macro, write_macros

ITALIAN_MONTH = {
    "gen": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "mag": 5,
    "giu": 6,
    "lug": 7,
    "ago": 8,
    "set": 9,
    "ott": 10,
    "nov": 11,
    "dic": 12,
}


def parse_mth(values):
    # Dates arrive as Excel dates, serial numbers, or Italian text ("31-gen-85").
    out = []
    for value in values:
        if pd.isna(value):
            out.append(pd.NaT)
            continue
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            out.append(pd.Timestamp(value).normalize())
            continue
        text = str(value).strip()
        if re.fullmatch(r"\d+", text):
            out.append(pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(text)))
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", text):
            out.append(pd.Timestamp(text[:10]))
            continue
        day, month, year = text.split("-")
        year = int(year)
        year = 2000 + year if year < 50 else 1900 + year
        out.append(pd.Timestamp(year=year, month=ITALIAN_MONTH[month], day=int(day)))
    return pd.to_datetime(out)


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def tex_vars(label):
    return re.sub(r"([A-Z]+)", r"\\texttt{\1}", label)


def conf(phat, R, alpha):
    up = np.asarray(R) > 0
    predicted_up = np.asarray(phat) > alpha
    return np.array(
        [
            np.mean(predicted_up),
            np.mean(predicted_up[up]),
            np.mean(~predicted_up[~up]),
        ]
    )


def main():
    clear_macros()

    dat = pd.read_excel("data/data_coursework1_Q2__2_.xls", sheet_name="final_m")
    dat["date"] = parse_mth(dat["mth"])
    if dat["date"].isna().any():
        raise AssertionError("A month date could not be parsed")
    ym = dat["date"].dt.year * 12 + dat["date"].dt.month
    if not np.all(np.diff(ym.to_numpy()) == 1):
        raise AssertionError("Dates are not consecutive months")
    n = len(dat)

    predictors = [
        "ESR",
        "USR",
        "EIF",
        "ERECB",
        "EFX",
        "EDY",
        "ESP",
        "UIF",
        "URR",
        "UDY",
        "UOIL",
    ]
    P = dat[predictors]
    display(dat.head())

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    t0 = int(np.flatnonzero((dat["date"].dt.year == 1994) & (dat["date"].dt.month == 12))[0])
    out = {}

    for target in ("ESR", "USR"):
        R = dat[target].to_numpy(dtype=float)
        Y = (R > 0).astype(float)

        srch = search_subsets(Y, R, P, alpha=0.5)
        bench = float(srch.loc[srch["mask"] == 0, "Z"].iloc[0])
        best = srch.iloc[0]
        cols = np.flatnonzero((int(best["mask"]) & (1 << np.arange(P.shape[1]))) > 0)
        fit = lpm_fit(Y, P, cols)
        Rn = R[1:]

        sc = alpha_scan(fit["phat"], Rn)
        a_star = float(np.mean(sc["intervals"][0]))

        pt05 = pt_test(fit["phat"], Rn, 0.5)
        oos_bench = oos_hit(Y, R, P, t0, cols=np.array([], dtype=int))
        oos_fixed = oos_hit(Y, R, P, t0, cols=cols)
        oos_sel = oos_hit(Y, R, P, t0, cols=None)

        def pt_oos(result):
            if np.unique(result["phat"] > 0.5).size < 2:
                return pd.Series({"stat": np.nan, "p": np.nan})
            return pt_test(result["phat"], result["R"], 0.5)

        out[target] = {
            "R": R,
            "Y": Y,
            "srch": srch,
            "bench": bench,
            "best": best,
            "cols": cols,
            "f": fit,
            "sc": sc,
            "a_star": a_star,
            "pt05": pt05,
            "pt_fixed": pt_oos(oos_fixed),
            "pt_sel": pt_oos(oos_sel),
            "up_sel": float(np.mean(oos_sel["phat"] > 0.5)),
            "Z0": hit_ratio(fit["phat"], Rn, 1e-9),
            "Z1": hit_ratio(fit["phat"], Rn, 1 - 1e-9),
            "Z45": hit_ratio(fit["phat"], Rn, 0.45),
            "Z55": hit_ratio(fit["phat"], Rn, 0.55),
            "oos": pd.Series(
                {
                    "bench": oos_bench["Z"],
                    "fixed": oos_fixed["Z"],
                    "sel": oos_sel["Z"],
                }
            ),
            "n_oos": oos_bench["n"],
            "chosen": oos_sel["chosen"],
        }

        intervals = " ".join(
            f"[{lo:.4f}, {hi:.4f})" for lo, hi in sc["intervals"]
        )
        print(
            f"\n== {target}: share up = {Y[1:].mean():.4f} | benchmark Z = {bench:.4f} | "
            f"best Z(1/2) = {best['Z']:.4f}, k = {int(best['k'])}: {best['vars']}"
        )
        print(
            f"   ties at max: {int((srch['Z'] == best['Z']).sum())} | "
            f"subsets beating benchmark: {int((srch['Z'] > bench).sum())} of {len(srch)}"
        )
        print(
            f"   Z(alpha) max = {sc['Zmax']:.4f} on {intervals} | "
            f"PT = {pt05['stat']:.3f} (p {pt05['p']:.4f})"
        )
        print(
            f"   OOS ({oos_bench['n']} forecasts): bench {oos_bench['Z']:.4f}, "
            f"fixed {oos_fixed['Z']:.4f}, reselected {oos_sel['Z']:.4f}"
        )

    labs = {"ESR": "EMU (ESR)", "USR": "US (USR)"}
    top_tab = [
        "\\begin{tabular}{rp{8.6cm}rrr}",
        "\\toprule",
        "Rank & Predictors in $X_t$ (plus a constant) & $k$ & $Z(1/2)$ & $\\bar R^2$ \\\\",
    ]
    for target in out:
        s = out[target]["srch"]
        top_tab.append("\\midrule")
        top_tab.append(
            "\\multicolumn{5}{l}{\\textit{$R_t$ = "
            f"{labs[target]}; constant only: $Z(1/2)={fmt(out[target]['bench'], 4)}$}}}} \\\\"
        )
        for rank in range(10):
            top_tab.append(
                f"{rank + 1} & {s.loc[rank, 'vars']} & {int(s.loc[rank, 'k'])} & "
                f"{fmt(s.loc[rank, 'Z'], 4)} & {fmt(s.loc[rank, 'R2adj'], 4)} \\\\"
            )
    top_tab.extend(["\\bottomrule", "\\end{tabular}"])
    write_lines("results/q2_top10.tex", top_tab)

    for target in out:
        fit = out[target]["f"]["fit"]
        rse = np.sqrt(np.diag(np.asarray(vcov_white(fit), dtype=float)))
        names = []
        for name in fit["beta"].index:
            if name == "const":
                names.append("constant")
            else:
                names.append(f"\\texttt{{{name}}}$_t$")
        rows = [
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Regressor & Estimate & Std.\\ error & $t$-stat & $p$-value & robust $t$ \\\\",
            "\\midrule",
        ]
        for i, name in enumerate(names):
            rows.append(
                f"{name} & {fmt(fit['beta'].iloc[i])} & {fmt(fit['se'].iloc[i])} & "
                f"{fmt(fit['t'].iloc[i], 3)} & {fmtp(fit['p'].iloc[i])} & "
                f"{fmt(fit['beta'].iloc[i] / rse[i], 3)} \\\\"
            )
        phat = out[target]["f"]["phat"]
        rows.extend(
            [
                "\\midrule",
                (
                    "\\multicolumn{6}{l}{\\footnotesize "
                    f"$n-1={fit['T']}$, $R^2={fmt(fit['R2'], 4)}$, "
                    f"$\\bar R^2={fmt(fit['R2adj'], 4)}$; fitted probabilities in "
                    f"$[{fmt(phat.min(), 3)},\\,{fmt(phat.max(), 3)}]$}} \\\\"
                ),
                "\\bottomrule",
                "\\end{tabular}",
            ]
        )
        write_lines(f"results/q2_coef_{target}.tex", rows)

    def iv(result):
        parts = [
            f"$[{fmt(lo, 3)},\\,{fmt(hi, 3)})$" for lo, hi in result["sc"]["intervals"]
        ]
        return ", ".join(parts)

    def row(label, getter):
        return f"{label} & {getter(out['ESR'])} & {getter(out['USR'])} \\\\"

    def pt_cell(result, key):
        stat = result[key]["stat"]
        if not np.isfinite(stat):
            return "n.a."
        return f"{fmt(stat, 2)} ({fmtp(result[key]['p'], 3)})"

    hit_tab = [
        "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{6.4cm}>{\\centering\\arraybackslash}p{3.4cm}>{\\centering\\arraybackslash}p{3.4cm}}",
        "\\toprule",
        f"& {labs['ESR']} & {labs['USR']} \\\\",
        "\\midrule",
        row("Share of positive months, $t=2,\\dots,n$", lambda o: fmt(np.mean(o["Y"][1:]), 4)),
        row("$Z(1/2)$, constant only (always ``up'')", lambda o: fmt(o["bench"], 4)),
        row("$Z(1/2)$, selected predictors", lambda o: fmt(o["best"]["Z"], 4)),
        row(
            "\\quad hit rate in up / down months",
            lambda o: (
                lambda cc: f"{fmt(cc[1], 3)} / {fmt(cc[2], 3)}"
            )(conf(o["f"]["phat"], o["R"][1:], 0.5)),
        ),
        row(
            "\\quad Pesaran--Timmermann test ($p$-value)",
            lambda o: f"{fmt(o['pt05']['stat'], 2)} ({fmtp(o['pt05']['p'], 3)})",
        ),
        "\\midrule",
        row("$\\max_\\alpha Z(\\alpha)$", lambda o: fmt(o["sc"]["Zmax"], 4)),
        row("\\quad attained for $\\alpha\\in$", iv),
        row(
            "\\quad hit rate in up / down months",
            lambda o: (
                lambda cc: f"{fmt(cc[1], 3)} / {fmt(cc[2], 3)}"
            )(conf(o["f"]["phat"], o["R"][1:], o["a_star"])),
        ),
        "\\midrule",
        row(
            f"Out of sample, 1995--2009 ({out['ESR']['n_oos']} months):",
            lambda o: "",
        ),
        row("\\quad constant only", lambda o: fmt(o["oos"]["bench"], 4)),
        row("\\quad selected predictors, fixed", lambda o: fmt(o["oos"]["fixed"], 4)),
        row("\\quad predictors re-selected each month", lambda o: fmt(o["oos"]["sel"], 4)),
        row("\\quad PT test, fixed set ($p$-value)", lambda o: pt_cell(o, "pt_fixed")),
        row("\\quad PT test, re-selected ($p$-value)", lambda o: pt_cell(o, "pt_sel")),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    write_lines("results/q2_hits.tex", hit_tab)

    last = dat["date"].iloc[n - 1]
    set_macro("qtwoN", n)
    set_macro("qtwoNm", n - 1)
    set_macro("qtwoK", P.shape[1])
    set_macro("qtwoNsub", 2 ** P.shape[1])
    set_macro("qtwoNoos", out["ESR"]["n_oos"])
    set_macro("qtwoOneHit", fmt(1 / (n - 1), 4))
    set_macro("qtwoLastDate", f"{last.day:02d} {last.strftime('%B')} {last.year}")
    for target, tg in (("ESR", "E"), ("USR", "U")):
        o = out[target]

        def put(name, value, tg=tg):
            set_macro(f"qtwo{tg}{name}", value)

        put("Share", fmt(np.mean(o["Y"][1:]), 4))
        put("Bench", fmt(o["bench"], 4))
        put("Best", fmt(o["best"]["Z"], 4))
        put("Vars", tex_vars(o["best"]["vars"]))
        put("Kbest", int(o["best"]["k"]))
        put("Ties", int((o["srch"]["Z"] == o["best"]["Z"]).sum()))
        put("Beat", int((o["srch"]["Z"] > o["bench"]).sum()))
        put("Gain", fmt(100 * (o["best"]["Z"] - o["bench"]), 1))
        put("Zmax", fmt(o["sc"]["Zmax"], 4))
        put("Alo", fmt(o["sc"]["intervals"][0, 0], 3))
        put("Ahi", fmt(o["sc"]["intervals"][0, 1], 3))
        put("Nint", o["sc"]["intervals"].shape[0])
        put("PT", fmt(o["pt05"]["stat"], 2))
        put("PTp", fmtp(o["pt05"]["p"], 3))
        put("OosB", fmt(o["oos"]["bench"], 4))
        put("OosF", fmt(o["oos"]["fixed"], 4))
        put("OosS", fmt(o["oos"]["sel"], 4))
        put("Pmin", fmt(np.min(o["f"]["phat"]), 3))
        put("Pmax", fmt(np.max(o["f"]["phat"]), 3))
        put("Rsq", fmt(o["f"]["fit"]["R2"], 3))
        put("Nchosen", len(set(o["chosen"])))
        put("Median", fmt(float(np.median(o["srch"]["Z"])), 4))
        put("BestPct", fmt(100 * o["best"]["Z"], 0))
        put("Astar", fmt(o["a_star"], 3))
        put("Zzero", fmt(o["Z0"], 3))
        put("Zone", fmt(o["Z1"], 3))
        put("Zff", fmt(o["Z45"], 3))
        put("Zfiftyfive", fmt(o["Z55"], 3))
        put("PTfix", fmt(o["pt_fixed"]["stat"], 2))
        put("PTfixp", fmtp(o["pt_fixed"]["p"], 3))
        if np.isfinite(o["pt_sel"]["stat"]):
            put("PTsel", fmt(o["pt_sel"]["stat"], 2))
            put("PTselp", fmtp(o["pt_sel"]["p"], 3))
        else:
            put("PTsel", "n.a.")
            put("PTselp", "n.a.")
        put("UpSel", fmt(100 * o["up_sel"], 0))
        put("Nout", int(np.sum((o["f"]["phat"] <= 0) | (o["f"]["phat"] >= 1))))
    write_macros("results/q2_macros.tex")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, target in zip(axes, out):
        o = out[target]
        breaks = (np.arange(120, 231, 2) + 0.5) / (n - 1)
        counts, edges = np.histogram(o["srch"]["Z"], bins=breaks)
        top = counts.max()
        ax.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color="#CCCCCC",
            edgecolor="white",
        )
        ax.set_xlim(0.54, 0.72)
        ax.set_ylim(0, 1.4 * top)
        bench_line, = ax.plot(
            [o["bench"], o["bench"]], [0, 1.08 * top], ls="--", lw=2, c="black"
        )
        best_line, = ax.plot(
            [o["best"]["Z"], o["best"]["Z"]], [0, 1.08 * top], ls="-", lw=2, c="#3A5FCD"
        )
        ax.set_title(labs[target])
        ax.set_xlabel("Z(1/2)")
        ax.set_ylabel("Number of combinations")
        ax.legend(
            [bench_line, best_line],
            ["constant only", "best"],
            frameon=False,
            loc="upper left",
            ncol=2,
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig("figures/q2_search.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    grid = np.linspace(0.001, 0.999, 999)
    for ax, target in zip(axes, out):
        o = out[target]
        zg = np.array([hit_ratio(o["f"]["phat"], o["R"][1:], a) for a in grid])
        ax.step(grid, zg, where="post", lw=1.5, c="black")
        half_line = ax.axvline(0.5, ls=":", c="black")
        bench_line = ax.axhline(o["bench"], ls="--", c="black")
        mids = o["sc"]["intervals"].mean(axis=1)
        maximum = ax.scatter(
            mids,
            np.full(len(mids), o["sc"]["Zmax"]),
            marker="v",
            c="#3A5FCD",
            s=30,
            zorder=3,
        )
        ax.set_ylim(0.35, 0.75)
        ax.set_title(labs[target])
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$Z(\alpha)$")
        ax.legend(
            [bench_line, half_line, maximum],
            ["constant only", r"$\alpha = 1/2$", "maximum"],
            frameon=False,
            loc="lower left",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig("figures/q2_zalpha.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7, 4.6))
    for ax, target in zip(axes, out):
        o = out[target]
        dates = dat["date"].iloc[1:].to_numpy()
        phat = o["f"]["phat"]
        ax.plot(dates, phat, lw=1.3, c="black")
        ax.axhline(0.5, ls=":", c="black")
        ax.axhline(0, color="#B3B3B3")
        ax.axhline(1, color="#B3B3B3")
        ax.set_ylim(min(0, phat.min()), max(1, phat.max()))
        up = o["R"][1:] > 0
        y_top = ax.get_ylim()[1]
        y_bot = ax.get_ylim()[0]
        ax.plot(dates[up], np.full(up.sum(), y_top), "|", c="#3A5FCD", ms=4)
        ax.plot(dates[~up], np.full((~up).sum(), y_bot), "|", c="#666666", ms=4)
        ax.set_title(labs[target])
        ax.set_ylabel(r"$\hat{P}(R_t > 0 \mid X_{t-1})$")
    fig.tight_layout()
    fig.savefig("figures/q2_phat.pdf")
    plt.close(fig)

    return out


if __name__ == "__main__":
    main()
