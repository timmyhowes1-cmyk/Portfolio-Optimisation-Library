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

The library has five layers that must be composed in order:

### 1. Data layer (`src/data/`)
- `data_loader.py` — fetches S&P 500 prices (yfinance), Fama-French 5-factor + momentum returns (pandas_datareader), float-adjusted market caps, and sector/region metadata. All loaders share a `_load_or_fetch` cache helper that reads from `src/data/*.pkl` and auto-refreshes after 5 days.
- `pipeline.py` — entry point: `load_data()` assembles the loaders and returns `(stock_returns, factor_returns, bmk_weights, sector_region)`.

### 2. Configuration (`src/config.py`)
`build_params(sector_region, bmk_weights)` returns the full params dict. Internally split into two concerns:
- **Policy** (`get_exclusions`, `get_industry_policy_bounds`) — ESG/SRI exclusion lists and industry bound overrides; change these when investment policy changes.
- **Calibration** — numeric parameters (`factor_halflife`, `stock_halflife`, `idio_halflife`, TE budget, turnover cap, etc.); change these when tuning the model.

Key param keys: `factor_halflife`, `stock_halflife`, `idio_halflife`, `sector_tolerance`, `industry_tolerance`, `exclusions`, `stock_active_weight`, `stock_active_weight_multiple`, `n_effective_stocks`, `max_te`, `max_turnover`.

### 3. Risk model (`src/risk/risk_model.py` — `EquityRiskModel`)
Built incrementally via three method calls:
- `add_stock_covariance(stock_returns, halflife=...)` — EWMA stock covariance and expected returns
- `add_factor_covariance(factor_returns, halflife=...)` — EWMA factor covariance
- `add_factor_exposure(stock_returns, factor_returns, halflife=..., idio_halflife=...)` — WLS betas; idiosyncratic volatility from EWMA variance of factor model residuals `ε = r - Xβ`

EWMA helpers (`ewma_cov`, `ewma_mean`, `ewma_var`, `ewma_weights`, `ewma_weighted`) live in `src/risk/ewma.py` as module-level functions, not methods on `EquityRiskModel`. The covariance implementation uses pairwise joint weights so the result is symmetric even when stocks have different NaN patterns (e.g. recent IPOs).

Idio risk uses residuals rather than `(1 - R²) × stock_variance`, so `factor_halflife` and `idio_halflife` can differ without creating an inconsistent decomposition. `add_factor_exposure` always triggers idio risk computation and stores residuals as `self.factor_residuals`.

### 4. Optimizers (`src/optimizers/`)
All optimizers inherit from `PortfolioMath` (`src/_portfolio_math.py`), which extracts the five precomputed arrays from the risk model (`_B`, `_F`, `_idio_sq`, `_mu`, `_bm_w`) and provides `portfolio_variance()`, `portfolio_return()`, `portfolio_volatility()`. This eliminates duplication across optimizer and metrics classes.

Two concrete base classes:
- **`CVXPYOptimizer`** (`cvxpy_base.py`) — uses CVXPY interior-point solver. Subclasses override `cvxpy_objective(self, w)`. Precomputes Cholesky form `F_sqrt_B` for SOCP. Used for all convex objectives.
- **`SLSQPOptimizer`** (`slsqp_base.py`) — uses scipy SLSQP. Subclasses override `objective(self, w, *args)` and optionally `objective_gradient`. Used for non-convex objectives (MaxSharpe, MaxDiversification).

| Class | Base | Objective |
|---|---|---|
| `MinVarianceOptimizer` | CVXPY | Minimise portfolio variance |
| `MeanVarianceOptimizer` | CVXPY | Mean-variance with `risk_aversion` param |
| `RiskParityOptimizerConvex` | CVXPY | Spinu convex formulation (equal risk contribution) |
| `MinTrackingErrorOptimizer` | CVXPY | Minimise TE vs benchmark |
| `MinTurnoverOptimizer` | CVXPY | Minimise one-way turnover |
| `MaxFactorTiltOptimizer` | CVXPY | Maximise exposure to a named factor |
| `MaxSharpeOptimizer` | SLSQP | Maximise Sharpe with `risk_free_rate` param |
| `MaxDiversificationOptimizer` | SLSQP | Maximise diversification ratio |

`ConstraintSet` (`src/optimizers/constraints.py`) is constructed with no arguments and built via a fluent interface. Each `add_*` method registers both an SLSQP constraint dict and a `_cvxpy_specs` entry so constraints work with both solver backends. Active weight and max-weight-multiple constraints are stored as box bounds for solver efficiency.

### 5. Performance (`src/performance/metrics.py` — `PortfolioMetrics`)
Also inherits `PortfolioMath`. Takes `(risk_model, weights, bm_weights)` and exposes individual metric methods plus `get_performance_metrics()`. All metrics use the factor model decomposition — tracking error and beta are ex-ante. Optimizer `get_performance_metrics()` and `get_holdings()` delegate here.

### Output (`src/output/charts.py` — `ExposureCharts`)
Takes weights and data directly — not an optimizer object:
```python
ExposureCharts(weights, bm_weights, risk_model, sector_region, output_dir='output')
```
Saves charts to `output/` by default (auto-created). `plot_all()` saves `all_exposures.png`; individual `plot_*` methods save named PNGs when called standalone.

### Public API
`src/__init__.py` explicitly re-exports: `load_data`, `load_factor_returns`, `load_sp500_prices`, `get_ff_shares`, `get_sector_region`, `download_prices`, `EquityRiskModel`. Optimizers from `src.optimizers`; performance from `src.performance`; charts from `src.output`.

### Canonical usage pattern
```python
from src.data import load_data
from src.config import build_params
from src.risk import EquityRiskModel
from src.optimizers import MinVarianceOptimizer, ConstraintSet
from src.performance import PortfolioMetrics
from src.output import ExposureCharts

stock_returns, factor_returns, bmk_weights, sector_region = load_data()
params = build_params(sector_region, bmk_weights)

rm = EquityRiskModel(tickers=bmk_weights.index.tolist(), sector_region=sector_region)
rm.add_stock_covariance(stock_returns, halflife=params["stock_halflife"])
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

ExposureCharts(opt.weights, bmk_weights, rm, sector_region).plot_all()
```