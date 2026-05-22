from src import load_sp500_prices, get_ff_shares, load_factor_returns, EquityRiskModel, get_sector_region
from src.optimizers import MaxFactorTiltOptimizer, ConstraintSet
import numpy as np

# Factor returns from Fama-French database, stock prices from yfinance
factor_returns = load_factor_returns().fillna(0)
prices = load_sp500_prices(start_date=factor_returns.index[0], end_date=factor_returns.index[-1]).dropna(axis=1, how='all').ffill()
stock_returns, factor_returns = prices.pct_change().dropna(axis=0, how='all').align(
    factor_returns,
    join='inner',  # Keep only common dates
    axis=0         # Align on rows (index)
)

# Get benchmark weights, sector and region data
sector_region = get_sector_region(tickers=stock_returns.columns.tolist())
ff_mcap = get_ff_shares(tickers=stock_returns.columns.tolist()) * prices.iloc[-1] # today's FF MCAP values
bmk_weights = (ff_mcap / ff_mcap.sum()).fillna(0)

# Build risk model
rm = EquityRiskModel(tickers=bmk_weights.index.tolist(), sector_region=sector_region)
rm.add_stock_covariance(stock_returns)
rm.add_factor_covariance(factor_returns)
rm.add_factor_exposure(stock_returns, factor_returns)

# Build constraint set
cs = (ConstraintSet()
      .add_grouping_constraint(sector_region['sector'].reindex(bmk_weights.index),
                               bm_weights=bmk_weights, tolerance=0.05)
      .add_grouping_constraint(sector_region['industry'].reindex(bmk_weights.index),
                               bm_weights=bmk_weights, tolerance=0.05)
      .add_active_weight_constraint(bmk_weights, bounds=0.02)
      .add_max_weight_multiple_constraint(bmk_weights, multiple=20)
      .add_effective_stocks_constraint(bounds=[1 / np.sum(bmk_weights ** 2), np.inf])
      .add_tracking_error_constraint(max_te=0.04, risk_model=rm, bm_weights=bmk_weights))

# Factor-tilt portfolio: maximise momentum exposure within a 4% annualised TE budget
opt = MaxFactorTiltOptimizer(risk_model=rm, bm_weights=bmk_weights, factor_name='MOM_Momentum')
opt.optimize(long_only=True, max_weight=0.07, constraints=cs)

metrics = {k: round(v, 3) if isinstance(v, float) else v for k, v in opt.get_performance_metrics().items()}
print(metrics)

print(opt.get_holdings().head(10).to_string(float_format=lambda x: f'{x:.3f}'))

