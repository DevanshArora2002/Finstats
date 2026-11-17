import numpy as np
import pandas as pd

from IPython.display import display
from scipy.optimize import minimize
from statsmodels.tools.numdiff import approx_hess


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


def estimate_cir_analytical(r):
    y = r[1:]
    x = r[:-1]
    n = len(y)

    x_mean = np.mean(x)
    z = x - x_mean
    X = np.column_stack((np.ones(n), z))

    # Weighted least squares (weights 1/x)
    w = 1.0 / x
    XtWX = X.T @ (X * w[:, None])
    XtWy = X.T @ (w * y)
    beta_hat = np.linalg.solve(XtWX, XtWy)

    # Extract estimates
    c_hat = beta_hat[0]
    phi_hat = beta_hat[1]
    a_hat = c_hat - phi_hat * x_mean

    # Recover CIR parameters
    mu_hat = a_hat / (1 - phi_hat)

    # Calculate residuals
    e_hat = y - X @ beta_hat

    # MLE of sigma squared
    sigma2_hat = np.sum((e_hat**2) / x) / n

    inputs = {
        "x": x,
        "y": y,
        "z": z,
        "n": n,
        "x_mean": x_mean,
        "X": X,
        "XtWX": XtWX,
        "c_hat": c_hat,
        "phi_hat": phi_hat,
    }
    return mu_hat, phi_hat, sigma2_hat, inputs


# Concentrated Gaussian log-likelihood in (c, phi)
def negloglik_profile(params, x, y, z, n):
    c, phi = params

    e = y - c - phi * z
    Q = np.sum(e**2 / x)

    if Q <= 0:
        return np.inf

    return 0.5 * (
        n * np.log(2 * np.pi)
        + np.sum(np.log(x))
        + n * (1 + np.log(Q / n))
    )


def gradient_profile(params, x, y, z, n):
    c, phi = params

    e = y - c - phi * z
    Q = np.sum(e**2 / x)

    return -(n / Q) * np.array([
        np.sum(e / x),
        np.sum(e * z / x)
    ])


# Full negative log-likelihood in (c, phi, log_sigma2)
def full_negloglik(q, x, y, z, n):
    c, phi, log_sigma2 = q

    e = y - c - phi * z

    return 0.5 * (
        n * np.log(2 * np.pi)
        + np.sum(np.log(x))
        + n * log_sigma2
        + np.sum(e**2 / x) * np.exp(-log_sigma2)
    )


def numerical_mle(inputs):
    x = inputs["x"]
    y = inputs["y"]
    z = inputs["z"]
    n = inputs["n"]
    x_mean = inputs["x_mean"]
    X = inputs["X"]

    # Initial values from ordinary least squares on the centred design matrix
    initial = np.linalg.lstsq(X, y, rcond=None)[0]

    fit_mle = minimize(
        negloglik_profile,
        x0=initial,
        args=(x, y, z, n),
        jac=gradient_profile,
        method="BFGS",
        options={
            "gtol": 1e-7,
            "maxiter": 10000
        }
    )

    c_num, phi_num = fit_mle.x
    a_num = c_num - phi_num * x_mean
    mu_num = a_num / (1 - phi_num)
    e_num = y - c_num - phi_num * z
    sigma2_num = np.mean(e_num**2 / x)

    theta = pd.Series({
        "mu": mu_num,
        "phi": phi_num,
        "sigma2": sigma2_num
    })
    return theta, fit_mle, c_num, phi_num, sigma2_num


def closed_form_covariance(mu_hat, phi_hat, sigma2_hat, inputs):
    n = inputs["n"]
    x_mean = inputs["x_mean"]
    XtWX = inputs["XtWX"]
    c_hat = inputs["c_hat"]

    # Covariance matrix of (c, phi)
    V_beta = sigma2_hat * np.linalg.inv(XtWX)

    # Covariance matrix of (c, phi, sigma2)
    V_eta = np.zeros((3, 3))
    V_eta[:2, :2] = V_beta
    V_eta[2, 2] = 2 * sigma2_hat**2 / n

    # Jacobian of (mu, phi, sigma2) with respect to (c, phi, sigma2)
    J = np.array([
        [1 / (1 - phi_hat), (c_hat - x_mean) / (1 - phi_hat)**2, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    param_names = ["mu", "phi", "sigma2"]
    return pd.DataFrame(
        J @ V_eta @ J.T,
        index=param_names,
        columns=param_names
    )


def numerical_covariance(inputs, c_num, phi_num, sigma2_num):
    x = inputs["x"]
    y = inputs["y"]
    z = inputs["z"]
    n = inputs["n"]
    x_mean = inputs["x_mean"]

    q_hat = np.array([
        c_num,
        phi_num,
        np.log(sigma2_num)
    ])

    H = approx_hess(
        q_hat,
        lambda q: full_negloglik(q, x, y, z, n)
    )
    V_q = np.linalg.inv(H)

    # Jacobian of (mu, phi, sigma2) with respect to (c, phi, log_sigma2)
    J_num = np.array([
        [1 / (1 - phi_num), (c_num - x_mean) / (1 - phi_num)**2, 0],
        [0, 1, 0],
        [0, 0, sigma2_num]
    ])

    param_names = ["mu", "phi", "sigma2"]
    return pd.DataFrame(
        J_num @ V_q @ J_num.T,
        index=param_names,
        columns=param_names
    )


def report_country(name, rates):
    mu_hat, phi_hat, sigma2_hat, inputs = estimate_cir_analytical(rates)

    theta_analytical = pd.Series({
        "mu": mu_hat,
        "phi": phi_hat,
        "sigma2": sigma2_hat
    })

    theta_numerical, fit_mle, c_num, phi_num, sigma2_num = numerical_mle(inputs)
    V_closed = closed_form_covariance(mu_hat, phi_hat, sigma2_hat, inputs)
    V_optim = numerical_covariance(inputs, c_num, phi_num, sigma2_num)

    print(f"\n========== {name} CIR ==========")

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


us = report_country("US", r_us)
uk = report_country("UK", r_uk)

theta_cir_analytical_us = us["analytical"]
theta_cir_analytical_uk = uk["analytical"]
theta_cir_numerical_us = us["numerical"]
theta_cir_numerical_uk = uk["numerical"]
V_closed_us = us["V_closed"]
V_closed_uk = uk["V_closed"]
V_optim_us = us["V_optim"]
V_optim_uk = uk["V_optim"]
