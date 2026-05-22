# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the project

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the main script
python main.py
```

Dependencies (not in a requirements file — install manually if missing):
`yfinance`, `pandas_datareader`, `scikit-learn`, `scipy`, `numpy`, `pandas`

## Architecture

The library has three layers that must be composed in order:

### 1. Data layer (`src/data/data_loader.py`)
Fetches S&P 500 prices (yfinance), Fama-French 5-factor + momentum returns (pandas_datareader), float-adjusted market caps, and sector/region metadata. All loaders use `src/data/*.pkl` file caches that auto-refresh if older than 5 days.

### 2. Risk model (`src/risk/risk_model.py` — `EquityRiskModel`)
Built incrementally via three method calls:
- `add_stock_covariance(stock_returns)` — sets annualised covariance and historical expected returns
- `add_factor_covariance(factor_returns)` — sets annualised factor covariance
- `add_factor_exposure(stock_returns, factor_returns)` — OLS-regresses each stock on factors; stores betas, R², and automatically computes idiosyncratic volatility as `sqrt((1 - R²) × stock_variance)`

Both `factor_exposure` and `idio_risk` must be populated before optimizing; `add_factor_exposure` triggers `idio_risk` computation automatically when `stock_covariance` is already set.

### 3. Optimizers (`src/optimizers/`)
`BaseOptimizer` wraps `scipy.optimize.minimize` (SLSQP). Portfolio variance is computed via the factor decomposition `w'B'FB w + Σ wᵢ²σᵢ²(idio)`, not the full stock covariance matrix directly. Concrete subclasses override only `objective(self, w, *args)`:

| Class | Objective |
|---|---|
| `MinVarianceOptimizer` | Minimise portfolio variance |
| `MeanVarianceOptimizer` | Mean-variance with `risk_aversion` param |
| `MaxSharpeOptimizer` | Maximise Sharpe with `risk_free_rate` param |
| `RiskParityOptimizerConvex` | Spinu convex formulation (equal risk contribution) |
| `MinTrackingErrorOptimizer` | Minimise TE vs benchmark |
| `MaxDiversificationOptimizer` | Maximise diversification ratio |
| `MinTurnoverOptimizer` | Minimise one-way turnover vs supplied weights |

`ConstraintSet` (`src/optimizers/constraints.py`) builds SLSQP constraint dicts for grouping constraints (e.g. sector/region bounds relative to benchmark weights).

### Public API
`src/__init__.py` re-exports everything from `src.data` and `src.risk`. Optimizers are not re-exported at the top-level and must be imported from `src.optimizers` directly.

### Canonical usage pattern
```python
# 1. Load and align data
factor_returns = load_factor_returns().fillna(0)
prices = load_sp500_prices(start_date=..., end_date=...).ffill()
stock_returns, factor_returns = prices.pct_change().dropna(axis=0, how='all').align(
    factor_returns, join='inner', axis=0
)

# 2. Build risk model
rm = EquityRiskModel(tickers=stock_returns.columns.tolist(), sector_region=sector_region)
rm.add_stock_covariance(stock_returns)
rm.add_factor_covariance(factor_returns)
rm.add_factor_exposure(stock_returns, factor_returns)

# 3. Optimize
opt = MinVarianceOptimizer(risk_model=rm, bm_weights=bmk_weights)
opt.optimize(long_only=True, max_weight=0.05)
print(opt.get_performance_metrics())
```
