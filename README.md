# Continuous-Time Stochastic Wealth Simulator

A high-performance Monte Carlo simulation engine designed to model long-term portfolio growth under uncertainty. Built in Python, this backend leverages Stochastic Differential Equations (SDEs) to generate thousands of parallel wealth trajectories, transforming discrete financial inputs into continuous-time probabilistic forecasts.

This engine is designed as the quantitative backend for a React-based frontend, outputting sanitized, downsampled JSON payloads representing complex probability cones (Fan Charts).

## Key Features

* **Geometric Brownian Motion (GBM):** Models asset price behavior using continuous-time stochastic calculus rather than discrete binomial branches.
* **Vectorized Execution:** Utilizes NumPy's SIMD capabilities to compute 10,000+ independent market paths across 252 annual trading days in milliseconds, avoiding slow iterative loops.
* **Rigorous Macroeconomic Adjustments:** * Calculates real (inflation-adjusted) returns using the exact Fisher equation rather than linear approximations.
  * Dynamically computes blended portfolio drift (μ) and volatility (σ) across diverse asset classes.
  * Applies continuous capital gains tax drag to accurately reflect wealth capture.
* **Dynamic Cash Flow Modeling:** Accounts for age-dependent contribution timelines, rigid liquidity events (e.g., college tuition, down payments), and Safe Withdrawal Rates (SWR) during retirement.
* **Frontend-Ready Payload:** Downsamples 100+ million raw data points into a lightweight JSON object containing interquartile percentile boundaries and background path noise for smooth browser rendering.

## Mathematical Foundation

The core simulation loop utilizes the Euler-Maruyama method to approximate the solution to the Geometric Brownian Motion SDE:

dW_t = μ W_t dt + σ W_t dX_t

Where:
* W_t is the portfolio wealth at time t.
* μ is the expected real drift, adjusted via the Fisher Equation: r = (1 + i) / (1 + π) - 1.
* σ is the blended portfolio volatility.
* dX_t is the underlying Wiener process (Brownian motion) scaling with √(dt).

To prevent negative wealth while preserving log-normal properties, the discrete step update leverages Itô's Lemma:

W_{t+dt} = W_t exp((μ - σ²/2)dt + σ√(dt)Z)

where Z ~ N(0,1).

## 🛠️ Setup & Installation

**Prerequisites:** Python 3.8+

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Install dependencies:**
   *(It is recommended to use a virtual environment)*
   ```bash
   pip install numpy matplotlib
   ```

3. **Configure Inputs:**
   Modify the `input.json` file to adjust asset allocations, income paths, inflation expectations, and liquidity events.

## Usage

The engine can be executed directly via the terminal. To run the simulation and generate the `simulation_output.json` payload:

```bash
make compile
```
*You will be prompted to enter the number of Monte Carlo runs (e.g., 10000).*

To run the simulation and immediately visualize the output locally using Matplotlib:

```bash
make run
```

## Project Structure

* `wienerProcessActual.py`: The core SDE engine and matrix logic.
* `visualize_json.py`: A local Matplotlib viewing tool for debugging the JSON payload geometry.
* `input_loader.py`: Safely parses and structures raw JSON strings into executable Python lists/tuples.
* `input.json`: The configuration file acting as the mock database for user financial parameters.
* `simulation_output.json`: The final generated REST-style payload consumed by the frontend.

## ⚡ Future Optimizations
* Integration of Cholesky Decomposition to model covariance between specific asset classes (e.g., S&P 500 and Aggregate Bonds).
* Implementation of a dynamic risk-free rate (r_f) to model shifting macroeconomic regimes.
* FastAPI wrapper to expose the engine as a live HTTP endpoint for frontend client requests.