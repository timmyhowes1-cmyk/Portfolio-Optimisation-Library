# Portfolio Optimisation Library

A Python library for constructing and optimising equity portfolios against the S&P 500, using a Fama-French factor risk model and a suite of SLSQP-based optimisers.

## Overview

The library pulls live market data, builds a multi-factor risk model, and solves a constrained portfolio optimisation problem. Portfolio variance is decomposed into systematic (factor) and idiosyncratic components, avoiding direct inversion of the full stock covariance matrix.

## Architecture

```
src/
├── data/data_loader.py      # S&P 500 prices, FF factors, market caps, sector/region metadata
├── risk/risk_model.py       # EquityRiskModel — stock & factor covariance, OLS factor exposures
└── optimizers/
    ├── base.py              # BaseOptimizer wrapping scipy SLSQP
    ├── optimizers.py        # Concrete optimiser classes
    └── constraints.py       # ConstraintSet builder
```

**Data layer** fetches S&P 500 prices (yfinance) and Fama-French 5-factor + momentum returns (pandas_datareader). Results are cached as `.pkl` files under `src/data/` and auto-refreshed after 5 days.

**Risk model** (`EquityRiskModel`) is built in three steps: stock covariance → factor covariance → OLS factor exposures + idiosyncratic volatility.

**Optimisers** share a common base; each subclass overrides only the objective function:

| Class | Objective |
|---|---|
| `MinVarianceOptimizer` | Minimise portfolio variance |
| `MeanVarianceOptimizer` | Mean-variance (configurable risk aversion) |
| `MaxSharpeOptimizer` | Maximise Sharpe ratio |
| `RiskParityOptimizerConvex` | Equal risk contribution (Spinu formulation) |
| `MinTrackingErrorOptimizer` | Minimise tracking error vs benchmark |
| `MaxDiversificationOptimizer` | Maximise diversification ratio |
| `MinTurnoverOptimizer` | Minimise one-way turnover |
| `MaxFactorTiltOptimizer` | Maximise exposure to a chosen factor |

`ConstraintSet` adds grouping constraints (sector, industry), active weight bounds, max weight multiples, effective-stocks floors, and tracking error caps.

## Usage

```bash
source .venv/bin/activate
python main.py
```

Dependencies: `yfinance`, `pandas_datareader`, `scikit-learn`, `scipy`, `numpy`, `pandas`

### Minimal example

```python
from src import load_sp500_prices, load_factor_returns, EquityRiskModel, get_sector_region, get_ff_shares
from src.optimizers import MinVarianceOptimizer
import numpy as np

factor_returns = load_factor_returns().fillna(0)
prices = load_sp500_prices(start_date=factor_returns.index[0], end_date=factor_returns.index[-1]).ffill()
stock_returns, factor_returns = prices.pct_change().dropna(axis=0, how='all').align(factor_returns, join='inner', axis=0)

sector_region = get_sector_region(tickers=stock_returns.columns.tolist())
ff_mcap = get_ff_shares(tickers=stock_returns.columns.tolist()) * prices.iloc[-1]
bmk_weights = (ff_mcap / ff_mcap.sum()).fillna(0)

rm = EquityRiskModel(tickers=bmk_weights.index.tolist(), sector_region=sector_region)
rm.add_stock_covariance(stock_returns)
rm.add_factor_covariance(factor_returns)
rm.add_factor_exposure(stock_returns, factor_returns)

opt = MinVarianceOptimizer(risk_model=rm, bm_weights=bmk_weights)
opt.optimize(long_only=True, max_weight=0.05)
print(opt.get_performance_metrics())
print(opt.get_holdings().head(10))
```