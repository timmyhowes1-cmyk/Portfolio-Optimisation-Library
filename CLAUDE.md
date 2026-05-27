# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Architecture

The library has four layers that must be composed in order:

### 1. Data layer (`src/data/data_loader.py`)
Fetches S&P 500 prices (yfinance), Fama-French 5-factor + momentum returns (pandas_datareader), float-adjusted market caps, and sector/region metadata. All loaders share a `_load_or_fetch` cache helper that reads from `src/data/*.pkl` and auto-refreshes if older than 5 days. Entry point is `load_data()` in `data_read.py`, which returns `(stock_returns, factor_returns, bmk_weights, sector_region)`.

### 2. Risk model (`src/risk/risk_model.py` — `EquityRiskModel`)
Built incrementally via three method calls, all with optional `halflife` for EWMA decay:
- `add_stock_covariance(stock_returns, halflife=...)` — EWMA stock covariance and expected returns
- `add_factor_covariance(factor_returns, halflife=...)` — EWMA factor covariance
- `add_factor_exposure(stock_returns, factor_returns, halflife=..., idio_halflife=...)` — WLS betas; idiosyncratic volatility from EWMA variance of factor model residuals `ε = r - Xβ`

Idio risk uses residuals rather than `(1 - R²) × stock_variance`, so `factor_halflife` and `idio_halflife` can differ without creating an inconsistent decomposition. `add_factor_exposure` always triggers idio risk computation and stores residuals as `self.factor_residuals`.

### 3. Optimizers (`src/optimizers/`)
`BaseOptimizer` wraps `scipy.optimize.minimize` (SLSQP). Portfolio variance is computed via the factor decomposition `w'B'FBw + Σwᵢ²σᵢ²(idio)`. Concrete subclasses override only `objective(self, w, *args)`:

| Class | Objective |
|---|---|
| `MinVarianceOptimizer` | Minimise portfolio variance |
| `MeanVarianceOptimizer` | Mean-variance with `risk_aversion` param |
| `MaxSharpeOptimizer` | Maximise Sharpe with `risk_free_rate` param |
| `RiskParityOptimizerConvex` | Spinu convex formulation (equal risk contribution) |
| `MinTrackingErrorOptimizer` | Minimise TE vs benchmark |
| `MaxDiversificationOptimizer` | Maximise diversification ratio |
| `MinTurnoverOptimizer` | Minimise one-way turnover vs supplied weights |
| `MaxFactorTiltOptimizer` | Maximise exposure to a named factor |

`ConstraintSet` (`src/optimizers/constraints.py`) builds SLSQP constraint dicts via a fluent interface. Active weight and max-weight-multiple constraints are stored as box bounds rather than SLSQP inequality constraints for solver efficiency.

### 4. Performance (`src/performance/metrics.py` — `PortfolioMetrics`)
Standalone class that takes `(risk_model, weights, bm_weights)` and exposes individual metric methods plus `get_performance_metrics()`. All metrics use the factor model decomposition — tracking error and beta are ex-ante. `BaseOptimizer.get_performance_metrics()` and `get_holdings()` delegate here.

### Charts (`src/charts.py` — `ExposureCharts`)
Takes a completed optimizer and `sector_region`. Saves charts to `charts/` by default (auto-created). `plot_all()` saves `all_exposures.png`; individual `plot_*` methods save named PNGs when called standalone.

### Configuration (`parameters.py`)
`build_params(sector_region, bmk_weights)` returns the full params dict including `sector_tolerance` (a DataFrame of per-sector active weight bounds). No data imports — takes data as arguments.

### Public API
`src/__init__.py` re-exports everything from `src.data` and `src.risk`. Optimizers must be imported from `src.optimizers`; performance from `src.performance`.

### Canonical usage pattern
```python
from data_read import load_data
from parameters import build_params
from src.risk import EquityRiskModel
from src.optimizers import MinVarianceOptimizer, ConstraintSet
from src.performance import PortfolioMetrics
from src.charts import ExposureCharts

stock_returns, factor_returns, bmk_weights, sector_region = load_data()
params = build_params(sector_region, bmk_weights)

rm = EquityRiskModel(tickers=bmk_weights.index.tolist(), sector_region=sector_region)
rm.add_stock_covariance(stock_returns, halflife=params["idio_halflife"])
rm.add_factor_covariance(factor_returns, halflife=params["factor_halflife"])
rm.add_factor_exposure(stock_returns, factor_returns,
                       halflife=params["factor_halflife"],
                       idio_halflife=params["idio_halflife"])

cs = (ConstraintSet()
      .add_grouping_constraint(sector_region['sector'].reindex(bmk_weights.index),
                               bm_weights=bmk_weights, tolerance=params["sector_tolerance"])
      .add_tracking_error_constraint(max_te=params["max_te"], risk_model=rm, bm_weights=bmk_weights))

opt = MinVarianceOptimizer(risk_model=rm, bm_weights=bmk_weights)
opt.optimize(long_only=True, constraints=cs)
print(opt.get_performance_metrics())

ExposureCharts(opt, sector_region).plot_all()
```