# Financial Statistics Coursework

Python implementations of three estimation problems from the Financial Statistics assignment: a CAPM regression with hypothesis tests, a linear probability model for the sign of market returns, and maximum-likelihood estimates of the CIR and Vasicek short-rate models.

The full brief is in [`Assignment_1_Brief__3.pdf`](Assignment_1_Brief__3.pdf). All code is written in Python.

## Setup

Requires Python 3.12 or newer. The pinned file was built with Python 3.14, so use 3.14 if an install fails on an older interpreter.

```bash
git clone https://github.com/DevanshArora2002/Finstats.git
cd Finstats
python3 -m venv .venv
```

Activate the environment:

```bash
# macOS and Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Install the pinned dependencies from [`requirements.txt`](requirements.txt):

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

That file is a full freeze, so every machine gets the same versions of pandas, NumPy, SciPy, statsmodels, xlrd, and Jupyter. The `.venv` directory stays local and is listed in `.gitignore`.

Register the environment as a notebook kernel, then select **Finstats** for the notebooks in Cursor or Jupyter:

```bash
python -m ipykernel install --user --name finstats --display-name "Finstats"
```

After someone changes dependencies, pull and reinstall:

```bash
pip install -r requirements.txt
```

## Questions

| | Topic | Method | Data |
| --- | --- | --- | --- |
| 1 | CAPM and asymmetric market exposure | OLS, *F* test, *t* test | `data coursework Q1` |
| 2 | Probability of a positive market return | OLS linear probability model, hit ratio | `data coursework Q2` |
| 3 | Term structure of interest rates | Gaussian MLE, asymptotic covariance | `data coursework Q3` |

## Question 1 — CAPM

Let $r_{it}$ be the return on stock $i$, $r_{Mt}$ the equity market index return, and $r_{ft}$ the risk-free rate. Two models are estimated by OLS over the full sample.

**Baseline CAPM**

$$
r_{it} - r_{ft} = \alpha + \beta (r_{Mt} - r_{ft}) + u_t
$$

**Asymmetric specification**

$$
r_{it} - r_{ft} = \alpha + \beta_1 \bigl[D_t (r_{Mt} - r_{ft})\bigr] + \beta_2 \bigl[(1 - D_t)(r_{Mt} - r_{ft})\bigr] + \beta_3 (r_{Mt} - r_{ft})^2 + u_t
$$

where $D_t = 1$ if $r_{Mt} - r_{ft} > 0$ and $D_t = 0$ otherwise.

**Tests**

- $F$ test of $H_0: \beta_1 = \beta_2$
- $t$ test of $H_0: \alpha = 0$

## Question 2 — Probability of a positive return

Let $R_t$ be an equity market index return and define the sign indicator

$$
Y_t = \begin{cases} 1 & \text{if } R_t > 0 \\ 0 & \text{if } R_t \le 0 \end{cases}
$$

The conditional probability is estimated by the linear model

$$
Y_{t+1} = \beta' X_t + \varepsilon_t, \qquad t = 1, \ldots, n - 1
$$

so that $\widehat{P}(R_{t+1} > 0 \mid X_t) = \hat\beta' X_t$.

Forecast performance is the hit ratio at a threshold $\alpha \in (0, 1)$:

$$
Z_t(\alpha) = \begin{cases} 1 & \text{if } \widehat{P}(R_t > 0 \mid X_{t-1}) > \alpha \text{ and } R_t > 0 \\ 1 & \text{if } \widehat{P}(R_t > 0 \mid X_{t-1}) \le \alpha \text{ and } R_t \le 0 \\ 0 & \text{otherwise} \end{cases}
$$

$$
Z(\alpha) = \frac{1}{n - 1} \sum_{t=2}^{n} Z_t(\alpha)
$$

**Tasks**

1. Estimate the regression by OLS on the full sample.
2. At $\alpha = 1/2$, search combinations of predictors for the largest $Z(\alpha)$.
3. For that combination, find the $\alpha$ that maximises $Z(\alpha)$.

## Question 3 — CIR and Vasicek

The discrete-time CIR short rate is

$$
r_t = \mu(1 - \phi) + \phi\, r_{t-1} + r_{t-1}^{1/2}\, u_t, \qquad u_t \sim \mathrm{NID}(0, \sigma^2)
$$

with parameter vector $\theta = (\mu, \phi, \sigma^2)'$. Estimation uses the Gaussian log-likelihood

$$
\ell(\theta) = \sum_{t=2}^{T} \log f(r_t \mid r_{t-1}, \theta)
$$

$$
f(r_t \mid r_{t-1}, \theta) = \frac{1}{\sqrt{2\pi\, r_{t-1}\, \sigma^2}} \exp\!\left( -\frac{1}{2\, r_{t-1}\, \sigma^2} \bigl(r_t - \mu(1 - \phi) - \phi\, r_{t-1}\bigr)^2 \right)
$$

and the asymptotic covariance matrix of the MLE.

The Vasicek benchmark drops the state-dependent volatility:

$$
r_t = \mu(1 - \phi) + \phi\, r_{t-1} + u_t, \qquad u_t \sim \mathrm{NID}(0, \sigma^2)
$$

Both models are estimated on the US and UK one-month rates $r_t^{\mathrm{US}}$ and $r_t^{\mathrm{UK}}$, and the fit of CIR is compared with Vasicek.

## Data

The coursework workbooks are already in `data/`. `xlrd`, installed with the requirements above, is what lets pandas read these `.xls` files.

- `data/data_coursework1_Q1__1_.xls` — stock return, market return, risk-free rate
- `data/data_coursework1_Q2__2_.xls` — market return and predictors $X_t$
- `data/data_coursework_Q3__1_.xls` — US and UK one-month interest rates
