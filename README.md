# Portfolio Optimisation Library

A Python library for constructing and optimising equity portfolios against the S&P 500, using a factor risk model and a suite of CVXPY and SLSQP-based optimisers.

## Overview

The library pulls live market data, builds a multi-factor risk model, and solves a constrained portfolio optimisation problem. Convex objectives are solved with CVXPY's interior-point solver; non-convex objectives fall back to scipy SLSQP.

## Architecture

```
src/
├── _portfolio_math.py        # Shared mixin: precomputed risk arrays + variance/return/volatility
├── config.py                 # build_params() — policy (exclusions) and calibration (halflives, TE budget)
├── data/
│   ├── data_loader.py        # S&P 500 prices, FF factors, market caps, sector metadata
│   └── pipeline.py           # load_data() — assembles data layer into a single call
├── risk/
│   ├── ewma.py               # EWMA helpers: ewma_cov, ewma_mean, ewma_var, ewma_weights
│   └── risk_model.py         # EquityRiskModel — EWMA covariance, WLS factor exposures, residual idio risk
├── optimizers/
│   ├── slsqp_base.py         # BaseOptimizer (SLSQP) — non-convex objectives
│   ├── cvxpy_base.py         # CVXPYOptimizer — convex objectives via interior-point solver
│   ├── optimizers.py         # Concrete optimiser subclasses
│   └── constraints.py        # ConstraintSet — fluent constraint builder (SLSQP + CVXPY backends)
├── performance/
│   └── metrics.py            # PortfolioMetrics — ex-ante risk, return, and attribution
└── output/
    └── charts.py             # ExposureCharts — active sector, factor, industry, and stock charts
```

## Data

Prices and factor returns are fetched on first run and cached as `.pkl` files under `src/data/`, auto-refreshed after 5 days. Benchmark weights are float-adjusted market-cap weights derived from yfinance share data.

## Risk model

`EquityRiskModel` is built in three steps with independent halflife controls:

```python
rm.add_stock_covariance(stock_returns, halflife=params["stock_halflife"])
rm.add_factor_covariance(factor_returns, halflife=params["factor_halflife"])
rm.add_factor_exposure(stock_returns, factor_returns,
                       halflife=params["factor_halflife"],
                       idio_halflife=params["idio_halflife"])
```

Idiosyncratic risk is estimated from EWMA variance of factor model residuals `ε = r - Xβ`, so factor and idio halflives are independently tunable.

## Optimisers

| Class | Solver | Objective |
|---|---|---|
| `MinVarianceOptimizer` | CVXPY | Minimise portfolio variance |
| `MeanVarianceOptimizer` | CVXPY | Mean-variance (configurable risk aversion) |
| `RiskParityOptimizerConvex` | CVXPY | Equal risk contribution (Spinu formulation) |
| `MinTrackingErrorOptimizer` | CVXPY | Minimise tracking error vs benchmark |
| `MinTurnoverOptimizer` | CVXPY | Minimise one-way turnover |
| `MaxFactorTiltOptimizer` | CVXPY | Maximise exposure to a named factor |
| `MaxSharpeOptimizer` | SLSQP | Maximise Sharpe ratio |
| `MaxDiversificationOptimizer` | SLSQP | Maximise diversification ratio |

## Constraints

`ConstraintSet` is built via a fluent interface. Each constraint works for both CVXPY and SLSQP automatically.

| Method | Constraint |
|---|---|
| `add_exclusions` | Force stocks to zero weight |
| `add_grouping_constraint` | Sector/industry active weight bounds |
| `add_attribute_constraint` | Scalar attribute exposure bounds |
| `add_active_weight_constraint` | Per-stock active weight bounds (box bounds) |
| `add_max_weight_multiple_constraint` | Per-stock weight ≤ N × benchmark weight |
| `add_tracking_error_constraint` | Ex-ante TE cap |
| `add_turnover_constraint` | One-way turnover cap |
| `add_effective_stocks_constraint` | Effective number of stocks bounds |
| `add_quadratic_constraint` | General quadratic constraint |

## Configuration

`src/config.py` separates two concerns:

- **Policy** `get_exclusions` — ESG/SRI exclusion lists
- **Calibration** — numeric parameters passed to the risk model and optimiser (halflives, TE budget, turnover cap, weight bounds)

## Performance metrics

`PortfolioMetrics` can be used standalone to evaluate any set of weights:

```python
from src.performance import PortfolioMetrics
pm = PortfolioMetrics(risk_model, weights, bm_weights)
pm.get_performance_metrics()    # return, vol, Sharpe, beta, TE, turnover, positions
pm.portfolio_tracking_error(w)  # ex-ante TE
pm.portfolio_beta(w)            # factor-model beta vs benchmark
```

## Charts

`ExposureCharts` takes weights and data directly to visualise portfolio exposures:

```python
from src.output import ExposureCharts
ExposureCharts(weights, bm_weights, risk_model, sector_region).plot_all()
```

Charts are saved to `output/` by default.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
