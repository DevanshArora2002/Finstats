from CIR import main as cir_main
from Vasicek import main as vasicek_main
import numpy as np
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

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
    
if __name__ == "__main__":
    main()
