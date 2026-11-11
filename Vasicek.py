#!/usr/bin/env python
# coding: utf-8

# First we do data handling.

# In[1]:


import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.tools.numdiff import approx_hess

# Import Excel data
data = pd.read_excel("data_coursework_Q3__1_.xls")

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
y_us = r_us[1:]
x_us = r_us[:-1]

y_uk = r_uk[1:]
x_uk = r_uk[:-1]


# Find Vasicek Closed Form Solution for MLE 

# In[4]:


# Design matrix
X_us = np.column_stack((np.ones(len(x_us)), x_us))

# Estimate a and b
beta_hat_us = np.linalg.solve(X_us.T @ X_us, X_us.T @ y_us)

# Extract estimates
a_hat_us = beta_hat_us[0]
b_hat_us = beta_hat_us[1]

# Recover Vasicek parameters
phi_hat_us = b_hat_us
mu_hat_us = a_hat_us / (1 - b_hat_us)

# Calculate residuals
e_hat_us = y_us - X_us @ beta_hat_us

# MLE of sigma squared
n_us = len(y_us)
sigma2_hat_us = np.sum(e_hat_us**2) / n_us

# Display estimates
theta_vasicek_analytical_us = pd.Series({
    "mu": mu_hat_us,
    "phi": phi_hat_us,
    "sigma2": sigma2_hat_us
})

display(theta_vasicek_analytical_us)


# Find numerical solution for MLE

# In[2]:


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


# Numerical maximum likelihood
fit_mle = minimize(
    negloglik,
    x0=[np.mean(y_us), 0.9, np.var(y_us, ddof=1)],
    args=(y_us, x_us),
    method="L-BFGS-B",
    bounds=[(None, None), (None, None), (1e-10, None)]
)

# Extract MLE estimates
theta_vasicek_numerical_us = pd.Series({
    "mu": fit_mle.x[0],
    "phi": fit_mle.x[1],
    "sigma2": fit_mle.x[2]
})

display(theta_vasicek_numerical_us)

print("Convergence:", fit_mle.success)
print(fit_mle.message)


# Find closed form solution for asymptotic covariance matrix

# In[5]:


# Covariance matrix of (a, b)
V_beta_us = sigma2_hat_us * np.linalg.inv(X_us.T @ X_us)

# Covariance matrix of (a, b, sigma2)
V_eta_us = np.zeros((3, 3))

V_eta_us[:2, :2] = V_beta_us
V_eta_us[2, 2] = 2 * sigma2_hat_us**2 / n_us

# Jacobian of (mu, phi, sigma2) with respect to (a, b, sigma2)
J_us = np.array([
    [1 / (1 - b_hat_us), a_hat_us / (1 - b_hat_us)**2, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Full asymptotic covariance matrix
V_closed_us = J_us @ V_eta_us @ J_us.T

param_names = ["mu", "phi", "sigma2"]

V_closed_us = pd.DataFrame(
    V_closed_us,
    index=param_names,
    columns=param_names
)

display(V_closed_us)


# Find numerical solution for covariance matrix

# In[3]:


# Numerical Hessian evaluated at the MLE
H_vasicek_us = approx_hess(
    fit_mle.x,
    lambda theta: negloglik(theta, y_us, x_us)
)

# Full asymptotic covariance matrix
V_optim_us = np.linalg.inv(H_vasicek_us)

V_optim_us = pd.DataFrame(
    V_optim_us,
    index=["mu", "phi", "sigma2"],
    columns=["mu", "phi", "sigma2"]
)

display(V_optim_us)

