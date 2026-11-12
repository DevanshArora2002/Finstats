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

# Define Y and X vectors
y = r_us[1:]
x = r_us[:-1]
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

# Display estimates
theta_cir_analytical_us = pd.Series({
    "mu": mu_hat,
    "phi": phi_hat,
    "sigma2": sigma2_hat
})

display(theta_cir_analytical_us)

# Concentrated Gaussian log-likelihood in (c, phi)
def negloglik_profile(params):
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


def gradient_profile(params):
    c, phi = params

    e = y - c - phi * z
    Q = np.sum(e**2 / x)

    return -(n / Q) * np.array([
        np.sum(e / x),
        np.sum(e * z / x)
    ])


# Initial values from ordinary least squares
initial = np.linalg.lstsq(X, y, rcond=None)[0]

# Numerical maximum likelihood
fit_mle = minimize(
    negloglik_profile,
    x0=initial,
    jac=gradient_profile,
    method="BFGS",
    options={
        "gtol": 1e-7,
        "maxiter": 10000
    }
)

# Extract MLE estimates
c_num, phi_num = fit_mle.x
a_num = c_num - phi_num * x_mean
mu_num = a_num / (1 - phi_num)
e_num = y - c_num - phi_num * z
sigma2_num = np.mean(e_num**2 / x)

theta_cir_numerical_us = pd.Series({
    "mu": mu_num,
    "phi": phi_num,
    "sigma2": sigma2_num
})

display(theta_cir_numerical_us)

print("Convergence:", fit_mle.success)
print(fit_mle.message)


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

# Full asymptotic covariance matrix
V_closed_us = J @ V_eta @ J.T

param_names = ["mu", "phi", "sigma2"]

V_closed_us = pd.DataFrame(
    V_closed_us,
    index=param_names,
    columns=param_names
)

display(V_closed_us)


# Full negative log-likelihood in (c, phi, log_sigma2)
def full_negloglik(q):
    c, phi, log_sigma2 = q

    e = y - c - phi * z

    return 0.5 * (
        n * np.log(2 * np.pi)
        + np.sum(np.log(x))
        + n * log_sigma2
        + np.sum(e**2 / x) * np.exp(-log_sigma2)
    )


q_hat = np.array([
    c_num,
    phi_num,
    np.log(sigma2_num)
])

# Numerical Hessian evaluated at the MLE
H_cir_us = approx_hess(q_hat, full_negloglik)

V_q = np.linalg.inv(H_cir_us)

# Jacobian of (mu, phi, sigma2) with respect to (c, phi, log_sigma2)
J_num = np.array([
    [1 / (1 - phi_num), (c_num - x_mean) / (1 - phi_num)**2, 0],
    [0, 1, 0],
    [0, 0, sigma2_num]
])

# Full asymptotic covariance matrix
V_optim_us = J_num @ V_q @ J_num.T

V_optim_us = pd.DataFrame(
    V_optim_us,
    index=["mu", "phi", "sigma2"],
    columns=["mu", "phi", "sigma2"]
)

display(V_optim_us)
