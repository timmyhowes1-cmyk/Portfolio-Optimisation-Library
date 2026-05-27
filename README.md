# Portfolio Optimisation Library

A Python library for constructing and optimising equity portfolios against the S&P 500, using an EWMA Fama-French factor risk model and a suite of SLSQP-based optimisers.

## Overview

The library pulls live market data, builds a multi-factor risk model, and solves a constrained portfolio optimisation problem. Portfolio variance is decomposed into systematic (factor) and idiosyncratic components, avoiding direct inversion of the full stock covariance matrix.

## Architecture

```
src/
├── data/data_loader.py      # S&P 500 prices, FF factors, market caps, sector metadata
├── risk/risk_model.py       # EquityRiskModel — EWMA covariance, WLS factor exposures, residual idio risk
├── optimizers/
│   ├── base.py              # BaseOptimizer wrapping scipy SLSQP
│   ├── optimizers.py        # Concrete optimiser subclasses
│   └── constraints.py       # ConstraintSet builder
├── performance/metrics.py   # PortfolioMetrics — ex-ante risk, return, and attribution
└── charts.py                # ExposureCharts — active sector, factor, industry, and stock charts
```

**Data layer** fetches S&P 500 prices (yfinance) and Fama-French 5-factor + momentum returns (pandas_datareader). Results are cached as `.pkl` files under `src/data/` and auto-refreshed after 5 days.

**Risk model** (`EquityRiskModel`) is built in three steps with optional EWMA decay:

1. `add_stock_covariance(stock_returns, halflife=...)` — EWMA stock covariance and expected returns
2. `add_factor_covariance(factor_returns, halflife=...)` — EWMA factor covariance matrix
3. `add_factor_exposure(stock_returns, factor_returns, halflife=..., idio_halflife=...)` — WLS factor betas; idiosyncratic volatility estimated from EWMA variance of factor model residuals

Using different halflives for factor covariance and idiosyncratic risk is supported and internally consistent.

**Optimisers** share a common base; each subclass overrides only the objective function. Portfolio variance is computed via the factor decomposition `w'B'FBw + Σwᵢ²σᵢ²`:

| Class | Objective |
|---|---|
| `MinVarianceOptimizer` | Minimise portfolio variance |
| `MeanVarianceOptimizer` | Mean-variance (configurable risk aversion) |
| `MaxSharpeOptimizer` | Maximise Sharpe ratio |
| `RiskParityOptimizerConvex` | Equal risk contribution (Spinu formulation) |
| `MinTrackingErrorOptimizer` | Minimise tracking error vs benchmark |
| `MaxDiversificationOptimizer` | Maximise diversification ratio |
| `MinTurnoverOptimizer` | Minimise one-way turnover |
| `MaxFactorTiltOptimizer` | Maximise exposure to a named factor |

**`ConstraintSet`** builds SLSQP constraint dicts via a fluent interface:

| Method | Constraint |
|---|---|
| `add_grouping_constraint` | Sector/industry active weight bounds |
| `add_attribute_constraint` | Scalar attribute exposure bounds (absolute or active) |
| `add_active_weight_constraint` | Per-stock active weight bounds (box bounds) |
| `add_max_weight_multiple_constraint` | Per-stock weight ≤ N × benchmark weight |
| `add_tracking_error_constraint` | Ex-ante TE cap |
| `add_turnover_constraint` | One-way turnover cap |
| `add_effective_stocks_constraint` | Effective number of stocks bounds |
| `add_quadratic_constraint` | General quadratic constraint |

**`PortfolioMetrics`** can be used standalone to evaluate any set of weights without an optimizer:

```python
from src.performance import PortfolioMetrics
pm = PortfolioMetrics(risk_model, weights, bm_weights)
pm.get_performance_metrics()   # return, vol, Sharpe, beta, TE, turnover, positions
pm.portfolio_tracking_error(w) # ex-ante TE
pm.portfolio_beta(w)           # factor-model beta vs benchmark
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Minimal example

```python
from data_read import load_data
from src.risk import EquityRiskModel
from src.optimizers import MinVarianceOptimizer, ConstraintSet

stock_returns, factor_returns, bmk_weights, sector_region = load_data()

rm = EquityRiskModel(tickers=bmk_weights.index.tolist(), sector_region=sector_region)
rm.add_stock_covariance(stock_returns, halflife=126)
rm.add_factor_covariance(factor_returns, halflife=252)
rm.add_factor_exposure(stock_returns, factor_returns, halflife=252, idio_halflife=126)

cs = (ConstraintSet()
      .add_grouping_constraint(sector_region['sector'].reindex(bmk_weights.index),
                               bm_weights=bmk_weights, tolerance=0.05)
      .add_tracking_error_constraint(max_te=0.04, risk_model=rm, bm_weights=bmk_weights))

opt = MinVarianceOptimizer(risk_model=rm, bm_weights=bmk_weights)
opt.optimize(long_only=True, constraints=cs)
print(opt.get_performance_metrics())
print(opt.get_holdings().head(10))
```