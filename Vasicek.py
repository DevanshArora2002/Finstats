import numpy as np
from IPython.display import display
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.tools.numdiff import approx_hess


# Negative conditional Gaussian log-likelihood
def negloglik(theta, y, x):
    mu = theta[0]
    phi = theta[1]
    sigma2 = theta[2]

    if sigma2 <= 0:
        return np.inf

    conditional_mean = mu * (1 - phi) + phi * x

    return -np.sum(
        norm.logpdf(
            y,
            loc=conditional_mean,
            scale=np.sqrt(sigma2)
        )
    )


def estimate_vasicek_analytical(y, x):
    # Design matrix
    X = np.column_stack((np.ones(len(x)), x))

    # Estimate a and b
    beta_hat = np.linalg.solve(X.T @ X, X.T @ y)

    # Extract estimates
    a_hat = beta_hat[0]
    b_hat = beta_hat[1]

    # Recover Vasicek parameters
    phi_hat = b_hat
    mu_hat = a_hat / (1 - b_hat)

    # Calculate residuals
    e_hat = y - X @ beta_hat

    # MLE of sigma squared
    n = len(y)
    sigma2_hat = np.sum(e_hat**2) / n

    theta = pd.Series({
        "mu": mu_hat,
        "phi": phi_hat,
        "sigma2": sigma2_hat
    })
    return theta, a_hat, b_hat, sigma2_hat, n, X


def numerical_mle(y, x):
    fit_mle = minimize(
        negloglik,
        x0=[np.mean(y), 0.9, np.var(y, ddof=1)],
        args=(y, x),
        method="L-BFGS-B",
        bounds=[(None, None), (None, None), (1e-10, None)]
    )

    theta = pd.Series({
        "mu": fit_mle.x[0],
        "phi": fit_mle.x[1],
        "sigma2": fit_mle.x[2]
    })
    return theta, fit_mle


def closed_form_covariance(a_hat, b_hat, sigma2_hat, n, X):
    # Covariance matrix of (a, b)
    V_beta = sigma2_hat * np.linalg.inv(X.T @ X)

    # Covariance matrix of (a, b, sigma2)
    V_eta = np.zeros((3, 3))
    V_eta[:2, :2] = V_beta
    V_eta[2, 2] = 2 * sigma2_hat**2 / n

    # Jacobian of (mu, phi, sigma2) with respect to (a, b, sigma2)
    J = np.array([
        [1 / (1 - b_hat), a_hat / (1 - b_hat)**2, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    param_names = ["mu", "phi", "sigma2"]
    return pd.DataFrame(
        J @ V_eta @ J.T,
        index=param_names,
        columns=param_names
    )


def numerical_covariance(fit_mle, y, x):
    H = approx_hess(
        fit_mle.x,
        lambda theta: negloglik(theta, y, x)
    )

    param_names = ["mu", "phi", "sigma2"]
    return pd.DataFrame(
        np.linalg.inv(H),
        index=param_names,
        columns=param_names
    )


def report_country(name, y, x):
    theta_analytical, a_hat, b_hat, sigma2_hat, n, X = estimate_vasicek_analytical(y, x)
    theta_numerical, fit_mle = numerical_mle(y, x)
    V_closed = closed_form_covariance(a_hat, b_hat, sigma2_hat, n, X)
    V_optim = numerical_covariance(fit_mle, y, x)

    print(f"\n========== {name} Vasicek ==========")

    print("\nAnalytical estimates")
    display(theta_analytical)

    print("\nNumerical MLE")
    display(theta_numerical)
    print("Convergence:", fit_mle.success)
    print(fit_mle.message)

    print("\nClosed-form asymptotic covariance")
    display(V_closed)

    print("\nNumerical asymptotic covariance")
    display(V_optim)

    return {
        "analytical": theta_analytical,
        "numerical": theta_numerical,
        "fit": fit_mle,
        "V_closed": V_closed,
        "V_optim": V_optim,
    }


def main():
    # Import Excel data
    data = pd.read_excel("data/data_coursework_Q3__1_.xls")

    display(data.head())

    # Rename columns
    data["r_US"] = data["T r_$"]
    data["r_UK"] = data["T r_£"]

    # Define interest rate vectors
    r_us = data["r_US"].to_numpy(dtype=float)
    r_uk = data["r_UK"].to_numpy(dtype=float)

    # Check number of observations
    T_us = len(r_us)
    T_uk = len(r_uk)

    print("US observations:", T_us)
    print("UK observations:", T_uk)

    y_us = r_us[1:]
    x_us = r_us[:-1]

    y_uk = r_uk[1:]
    x_uk = r_uk[:-1]

    us = report_country("US", y_us, x_us)
    uk = report_country("UK", y_uk, x_uk)

    theta_vasicek_analytical_us = us["analytical"]
    theta_vasicek_analytical_uk = uk["analytical"]
    theta_vasicek_numerical_us = us["numerical"]
    theta_vasicek_numerical_uk = uk["numerical"]
    V_closed_us = us["V_closed"]
    V_closed_uk = uk["V_closed"]
    V_optim_us = us["V_optim"]
    V_optim_uk = uk["V_optim"]

    return {
        "theta_vasicek_analytical_us": theta_vasicek_analytical_us,
        "theta_vasicek_analytical_uk": theta_vasicek_analytical_uk,
        "theta_vasicek_numerical_us": theta_vasicek_numerical_us,
        "theta_vasicek_numerical_uk": theta_vasicek_numerical_uk,
        "V_closed_us": V_closed_us,
        "V_closed_uk": V_closed_uk,
        "V_optim_us": V_optim_us,
        "V_optim_uk": V_optim_uk,
    }


if __name__ == "__main__":
    main()
