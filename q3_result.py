import os

from src.CIR import main as cir_main
from src.Vasicek import main as vasicek_main
import numpy as np
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from src.table_helpers import clear_macros, fmt, fmtp, set_macro, write_macros

from scipy.optimize import minimize
from scipy.stats import chi2
from statsmodels.tools.numdiff import approx_fprime, approx_hess

def free_negloglik(params, y, x):
    mu, phi, log_sigma2, gamma = params
    mean = mu * (1.0 - phi) + phi * x
    log_variance = log_sigma2 + gamma * np.log(x)
    variance = np.exp(log_variance)
    error = y - mean
    return 0.5 * np.sum(np.log(2.0 * np.pi * variance) + error**2 / variance)


def restricted_loglik(theta, y, x, gamma0):
    mu = float(theta["mu"])
    phi = float(theta["phi"])
    sigma2 = float(theta["sigma2"])
    mean = mu * (1.0 - phi) + phi * x
    variance = sigma2 * np.power(x, gamma0)
    error = y - mean
    return float(np.sum(-0.5 * (np.log(2.0 * np.pi * variance) + error**2 / variance)))


def fit_free_gamma(y, x, theta_vasicek, theta_cir):
    starts = []
    for theta, gamma0 in ((theta_vasicek, 0.0), (theta_cir, 1.0)):
        starts.append(
            np.array(
                [
                    float(theta["mu"]),
                    float(theta["phi"]),
                    np.log(float(theta["sigma2"])),
                    gamma0,
                ],
                dtype=np.float64,
            )
        )

    candidates = []
    for start in starts:
        fit = minimize(
            free_negloglik,
            start,
            args=(y, x),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-14},
        )
        candidates.append(fit)

    best = min(candidates, key=lambda fit: fit.fun)
    if not np.isfinite(best.fun):
        raise RuntimeError("Free-gamma fit returned a non-finite objective")
    return best, candidates


def score_and_hessian(params, y, x):
    score = approx_fprime(params, free_negloglik, args=(y, x), centered=True)
    hessian = approx_hess(params, free_negloglik, args=(y, x))
    return np.asarray(score, dtype=np.float64), np.asarray(hessian, dtype=np.float64)


def quadratic_form(score, hessian):
    solved = np.linalg.solve(hessian, score)
    return float(score @ solved)

def gaussian_loglik(error, variance):
    return float(np.sum(-0.5 * (np.log(2.0 * np.pi * variance) + error**2 / variance)))


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def decide(p):
    return "Reject" if p < 0.05 else "Do not reject"


def fit_model(theta, y, x, model):
    mu = float(theta["mu"])
    phi = float(theta["phi"])
    sigma2 = float(theta["sigma2"])
    fitted = mu * (1.0 - phi) + phi * x
    error = y - fitted
    if model == "CIR":
        variance = sigma2 * x
    elif model == "Vasicek":
        variance = np.full(y.shape, sigma2, dtype=np.float64)
    else:
        raise ValueError(model)
    return {
        "fitted": fitted,
        "error": error,
        "variance": variance,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "loglik": gaussian_loglik(error, variance),
    }



def main():
    clear_macros()

    results_cir = cir_main()
    results_vasicek = vasicek_main()


    theta_cir = {
        "US": results_cir["theta_cir_analytical_us"].astype(np.float64),
        "UK": results_cir["theta_cir_analytical_uk"].astype(np.float64),
    }
    theta_vasicek = {
        "US": results_vasicek["theta_vasicek_analytical_us"].astype(np.float64),
        "UK": results_vasicek["theta_vasicek_analytical_uk"].astype(np.float64),
    }
    V_cir = {
        "US": results_cir["V_closed_us"],
        "UK": results_cir["V_closed_uk"],
    }
    V_vasicek = {
        "US": results_vasicek["V_closed_us"],
        "UK": results_vasicek["V_closed_uk"],
    }

    rates = pd.read_excel("data/data_coursework_Q3__1_.xls")
    rates["date"] = pd.to_datetime(rates.iloc[:, 0])

    samples = {}
    for country, column in (("US", "T r_$"), ("UK", "T r_£")):
        level = rates[column].to_numpy(dtype=np.float64)
        samples[country] = pd.DataFrame(
            {
                "date": rates["date"].iloc[1:].to_numpy(),
                "y": level[1:],
                "x": level[:-1],
            }
        )
        assert len(samples[country]) == 313, len(samples[country])
        assert np.all(samples[country]["x"].to_numpy() > 0)

    print("Comparison sample: February 1978 through February 2004, spreadsheet percent units")
    for country in ("US", "UK"):
        sample = samples[country]
        print(
            f"{country}: n={len(sample)}, "
            f"{sample['date'].iloc[0].date()} to {sample['date'].iloc[-1].date()}, "
            f"min lag={sample['x'].min():.2f}"
        )
        print(country, "CIR analytical")
        display(theta_cir[country])
        print(country, "Vasicek analytical")
        display(theta_vasicek[country])

    fits = {}
    for country in ("US", "UK"):
        y = samples[country]["y"].to_numpy(dtype=np.float64)
        x = samples[country]["x"].to_numpy(dtype=np.float64)
        samples[country]["bin"] = pd.qcut(x, 3, labels=["low", "medium", "high"])
        fits[country] = {
            "CIR": fit_model(theta_cir[country], y, x, "CIR"),
            "Vasicek": fit_model(theta_vasicek[country], y, x, "Vasicek"),
        }

    fit_rows = []
    for country in ("US", "UK"):
        cir_ll = fits[country]["CIR"]["loglik"]
        vas_ll = fits[country]["Vasicek"]["loglik"]
        for model in ("CIR", "Vasicek"):
            fit_rows.append(
                {
                    "country": country,
                    "model": model,
                    "n": len(samples[country]),
                    "RMSE": fits[country][model]["rmse"],
                    "loglik": fits[country][model]["loglik"],
                }
            )
        fit_rows.append(
            {
                "country": country,
                "model": "CIR minus Vasicek",
                "n": len(samples[country]),
                "RMSE": np.nan,
                "loglik": cir_ll - vas_ll,
            }
        )

    fit_table = pd.DataFrame(fit_rows)
    for country in ("US", "UK"):
        print(f"\n{country} fit")
        display(
            fit_table.loc[fit_table["country"] == country, ["model", "n", "RMSE", "loglik"]]
            .set_index("model")
        )

    calibration_rows = []
    for country in ("US", "UK"):
        bins = samples[country]["bin"]
        for model in ("CIR", "Vasicek"):
            frame = pd.DataFrame(
                {
                    "bin": bins.to_numpy(),
                    "squared_error": fits[country][model]["error"] ** 2,
                    "fitted_variance": fits[country][model]["variance"],
                }
            )
            grouped = frame.groupby("bin", observed=True).agg(
                n=("squared_error", "size"),
                mean_squared_error=("squared_error", "mean"),
                mean_fitted_variance=("fitted_variance", "mean"),
            )
            for bin_name, row in grouped.iterrows():
                calibration_rows.append(
                    {
                        "country": country,
                        "model": model,
                        "bin": str(bin_name),
                        "n": int(row["n"]),
                        "mean_squared_error": float(row["mean_squared_error"]),
                        "mean_fitted_variance": float(row["mean_fitted_variance"]),
                    }
                )

    calibration = pd.DataFrame(calibration_rows)
    for country in ("US", "UK"):
        print(f"\n{country} variance calibration by lagged-rate tercile")
        display(
            calibration.loc[calibration["country"] == country]
            .drop(columns="country")
            .set_index(["model", "bin"])
        )
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )

    countries = ("US", "UK")
    models = ("CIR", "Vasicek")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for i, country in enumerate(countries):
        dates = samples[country]["date"]
        y = samples[country]["y"].to_numpy(dtype=np.float64)
        for j, model in enumerate(models):
            ax = axes[i, j]
            fitted = fits[country][model]["fitted"]
            band = 1.96 * np.sqrt(fits[country][model]["variance"])
            ax.fill_between(dates, fitted - band, fitted + band, color="C0", alpha=0.18, label="95% band")
            ax.plot(dates, y, color="0.2", lw=0.9, label=r"$r_t$")
            ax.plot(dates, fitted, color="C0", lw=1.1, label=r"$\hat r_t$")
            ax.set_title(f"{country} {model}")
            if i == 1:
                ax.set_xlabel("Date")
            if j == 0:
                ax.set_ylabel("Percent")
    axes[0, 0].legend(loc="upper right", frameon=False)
    fig.suptitle("One-step mean and 95% prediction band")
    fig.tight_layout()
    fig.savefig("figures/q3_one_step_mean_band.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for i, country in enumerate(countries):
        for j, model in enumerate(models):
            ax = axes[i, j]
            standardized = fits[country][model]["error"] / np.sqrt(fits[country][model]["variance"])
            stats.probplot(standardized, dist="norm", plot=ax)
            ax.set_title(f"{country} {model}")
            ax.get_lines()[0].set_markersize(4)
            ax.get_lines()[0].set_color("0.2")
    fig.suptitle(r"Normal Q–Q plots of $e_t/\sqrt{v_t}$")
    fig.tight_layout()
    fig.savefig("figures/q3_qq.pdf")
    plt.close(fig)


    test_rows = []
    free_fits = {}
    for country in ("US", "UK"):
        y = samples[country]["y"].to_numpy(dtype=np.float64)
        x = samples[country]["x"].to_numpy(dtype=np.float64)
        best, candidates = fit_free_gamma(y, x, theta_vasicek[country], theta_cir[country])
        free_fits[country] = best
        gamma_hat = float(best.x[3])
        ell_free = float(-best.fun)
        _, hessian_free = score_and_hessian(best.x, y, x)
        covariance = np.linalg.inv(hessian_free)
        gamma_variance = float(covariance[3, 3])
        if gamma_variance <= 0:
            raise RuntimeError(f"{country} free-gamma Hessian gave a non-positive variance")

        print(
            f"{country} free fit: success={best.success}, gamma={gamma_hat:.6f}, "
            f"loglik={ell_free:.6f}, message={best.message}"
        )

        for gamma0, theta in ((0.0, theta_vasicek[country]), (1.0, theta_cir[country])):
            ell_restricted = restricted_loglik(theta, y, x, gamma0)
            lr = 2.0 * (ell_free - ell_restricted)
            wald = (gamma_hat - gamma0) ** 2 / gamma_variance
            restricted_params = np.array(
                [
                    float(theta["mu"]),
                    float(theta["phi"]),
                    np.log(float(theta["sigma2"])),
                    gamma0,
                ],
                dtype=np.float64,
            )
            score, hessian = score_and_hessian(restricted_params, y, x)
            lm = quadratic_form(score, hessian)
            test_rows.append(
                {
                    "country": country,
                    "restriction": f"gamma={int(gamma0)}",
                    "gamma_hat": gamma_hat,
                    "gamma_se": float(np.sqrt(gamma_variance)),
                    "LR": lr,
                    "LR_p": float(chi2.sf(lr, 1)),
                    "Wald": wald,
                    "Wald_p": float(chi2.sf(wald, 1)),
                    "LM": lm,
                    "LM_p": float(chi2.sf(lm, 1)),
                    "loglik_free": ell_free,
                    "loglik_restricted": ell_restricted,
                }
            )

    test_table = pd.DataFrame(test_rows)
    if np.any(test_table["LR"] < -1e-8):
        raise RuntimeError("A likelihood-ratio statistic is negative")
    for country in ("US", "UK"):
        print(f"\n{country} tests of the variance power")
        display(
            test_table.loc[test_table["country"] == country]
            .drop(columns=["country", "loglik_free", "loglik_restricted"])
            .set_index("restriction")
        )

    params = ("mu", "phi", "sigma2")
    param_tex = {"mu": r"$\mu$", "phi": r"$\phi$", "sigma2": r"$\sigma^2$"}
    thetas = {"CIR": theta_cir, "Vasicek": theta_vasicek}
    covariances = {"CIR": V_cir, "Vasicek": V_vasicek}

    def se(model, country, name):
        variance = float(covariances[model][country].loc[name, name])
        return float(np.sqrt(variance))

    estimate_lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{CIR} & \multicolumn{3}{c}{Vasicek} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r" & $\mu$ & $\phi$ & $\sigma^2$ & $\mu$ & $\phi$ & $\sigma^2$ \\",
        r"\midrule",
    ]
    for country in ("US", "UK"):
        estimate_lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{country}}}}} \\")
        estimates = []
        errors = []
        for model in ("CIR", "Vasicek"):
            for name in params:
                estimates.append(fmt(thetas[model][country][name]))
                errors.append(fmt(se(model, country, name)))
        estimate_lines.append("Estimate & " + " & ".join(estimates) + r" \\")
        estimate_lines.append(r"Std.\ error & " + " & ".join(errors) + r" \\")
        if country == "US":
            estimate_lines.append(r"\midrule")
    estimate_lines.extend(
        [
            r"\midrule",
            (
                r"\multicolumn{7}{l}{\footnotesize Analytical estimates. "
                r"Standard errors are the square roots of the diagonal of the "
                r"closed-form asymptotic covariance.} \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write_lines("results/q3_estimates.tex", estimate_lines)

    covariance_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r" & $\mu$ & $\phi$ & $\sigma^2$ \\",
        r"\midrule",
    ]
    first_block = True
    for country in ("US", "UK"):
        for model in ("CIR", "Vasicek"):
            if not first_block:
                covariance_lines.append(r"\midrule")
            first_block = False
            matrix = covariances[model][country]
            covariance_lines.append(
                rf"\multicolumn{{4}}{{l}}{{\textit{{{country}, {model}}}}} \\"
            )
            for name in params:
                cells = " & ".join(fmt(float(matrix.loc[name, other]), 6) for other in params)
                covariance_lines.append(f"{param_tex[name]} & {cells} \\\\")
    covariance_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_lines("results/q3_covariance.tex", covariance_lines)

    fit_lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Country & Model & $n$ & RMSE & Log-likelihood \\",
        r"\midrule",
    ]
    for i, country in enumerate(("US", "UK")):
        if i:
            fit_lines.append(r"\midrule")
        first = True
        block = fit_table.loc[fit_table["country"] == country]
        for _, row in block.iterrows():
            country_cell = country if first else ""
            first = False
            if row["model"] == "CIR minus Vasicek":
                fit_lines.append(
                    f"{country_cell} & CIR $-$ Vasicek & & & {fmt(row['loglik'], 3)} \\\\"
                )
                continue
            fit_lines.append(
                f"{country_cell} & {row['model']} & {int(row['n'])} & "
                f"{fmt(row['RMSE'], 4)} & {fmt(row['loglik'], 3)} \\\\"
            )
    fit_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_lines("results/q3_fit.tex", fit_lines)

    tercile = {"low": "Low", "medium": "Medium", "high": "High"}
    calibration_lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Country & Model & Tercile of $r_{t-1}$ & $n$ & Mean squared error & Mean fitted variance \\",
        r"\midrule",
    ]
    for i, country in enumerate(("US", "UK")):
        if i:
            calibration_lines.append(r"\midrule")
        first = True
        block = calibration.loc[calibration["country"] == country]
        for model in ("CIR", "Vasicek"):
            for bin_name in ("low", "medium", "high"):
                row = block.loc[
                    (block["model"] == model) & (block["bin"] == bin_name)
                ].iloc[0]
                country_cell = country if first else ""
                first = False
                calibration_lines.append(
                    f"{country_cell} & {model} & {tercile[bin_name]} & {int(row['n'])} & "
                    f"{fmt(row['mean_squared_error'], 4)} & {fmt(row['mean_fitted_variance'], 4)} \\\\"
                )
    calibration_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_lines("results/q3_calibration.tex", calibration_lines)

    crit = float(chi2.ppf(0.95, 1))
    null_tex = {
        "gamma=0": r"$\gamma=0$ (Vasicek)",
        "gamma=1": r"$\gamma=1$ (CIR)",
    }
    test_lines = [
        r"\begin{tabular}{lllrrrl}",
        r"\toprule",
        r"Country & Null hypothesis & Test & Statistic & 5\% crit.\ value & $p$-value & Decision \\",
        r"\midrule",
    ]
    for i, country in enumerate(("US", "UK")):
        if i:
            test_lines.append(r"\midrule")
        block = test_table.loc[test_table["country"] == country]
        gamma_hat = float(block["gamma_hat"].iloc[0])
        gamma_se = float(block["gamma_se"].iloc[0])
        test_lines.append(
            r"\multicolumn{7}{l}{\footnotesize "
            + f"{country}: $\\hat\\gamma={fmt(gamma_hat)}$, s.e.\\ {fmt(gamma_se)}}} \\\\"
        )
        country_pending = True
        for _, row in block.iterrows():
            null_pending = True
            for label, stat_key, p_key in (
                ("LR", "LR", "LR_p"),
                ("Wald", "Wald", "Wald_p"),
                ("LM", "LM", "LM_p"),
            ):
                country_cell = country if country_pending else ""
                null_cell = null_tex[row["restriction"]] if null_pending else ""
                country_pending = False
                null_pending = False
                p_value = float(row[p_key])
                test_lines.append(
                    f"{country_cell} & {null_cell} & {label} & {fmt(row[stat_key], 3)} & "
                    f"{fmt(crit, 3)} & {fmtp(p_value)} & {decide(p_value)} \\\\"
                )
    test_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_lines("results/q3_tests.tex", test_lines)

    start = samples["US"]["date"].iloc[0]
    end = samples["US"]["date"].iloc[-1]
    set_macro("qthreeN", int(len(samples["US"])))
    set_macro("qthreeStart", f"{start.day:02d} {start.strftime('%B')} {start.year}")
    set_macro("qthreeEnd", f"{end.day:02d} {end.strftime('%B')} {end.year}")
    set_macro("qthreeCrit", fmt(crit, 3))
    country_key = {"US": "Us", "UK": "Uk"}
    model_key = {"CIR": "Cir", "Vasicek": "Vas"}
    param_key = {"mu": "Mu", "phi": "Phi", "sigma2": "Sig"}
    for country in ("US", "UK"):
        ck = country_key[country]
        for model in ("CIR", "Vasicek"):
            mk = model_key[model]
            for name in params:
                set_macro(
                    f"qthree{ck}{mk}{param_key[name]}",
                    fmt(thetas[model][country][name]),
                )
                set_macro(
                    f"qthree{ck}{mk}{param_key[name]}Se",
                    fmt(se(model, country, name)),
                )
            set_macro(f"qthree{ck}{mk}Rmse", fmt(fits[country][model]["rmse"], 4))
            set_macro(f"qthree{ck}{mk}Ll", fmt(fits[country][model]["loglik"], 3))
        diff = fits[country]["CIR"]["loglik"] - fits[country]["Vasicek"]["loglik"]
        set_macro(f"qthree{ck}LlDiff", fmt(diff, 3))
        block = test_table.loc[test_table["country"] == country]
        set_macro(f"qthree{ck}Gamma", fmt(float(block["gamma_hat"].iloc[0])))
        set_macro(f"qthree{ck}GammaSe", fmt(float(block["gamma_se"].iloc[0])))
        for restriction, tag in (("gamma=0", "G0"), ("gamma=1", "G1")):
            row = block.loc[block["restriction"] == restriction].iloc[0]
            for label, stat_key, p_key in (
                ("Lr", "LR", "LR_p"),
                ("Wald", "Wald", "Wald_p"),
                ("Lm", "LM", "LM_p"),
            ):
                set_macro(f"qthree{ck}{label}{tag}", fmt(row[stat_key], 3))
                set_macro(f"qthree{ck}{label}{tag}p", fmtp(float(row[p_key])))
    write_macros("results/q3_macros.tex")

if __name__ == "__main__":
    main()
