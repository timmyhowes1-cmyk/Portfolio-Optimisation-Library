import time
import pandas as pd
from src.data import load_data
from src.config import build_params
from src.risk import EquityRiskModel
from src.optimizers import MaxFactorTiltOptimizer, ConstraintSet
from src.performance import PortfolioMetrics
from src.output import ExposureCharts

stock_returns, factor_returns, bmk_weights, sector_region = load_data()
params = build_params(sector_region, bmk_weights)

# Build risk model
rm = EquityRiskModel(tickers=bmk_weights.index.tolist(), sector_region=sector_region)
rm.add_stock_covariance(stock_returns, halflife=params["stock_halflife"])
rm.add_factor_covariance(factor_returns, halflife=params["factor_halflife"])
rm.add_factor_exposure(stock_returns, factor_returns, halflife=params["factor_halflife"],
                       idio_halflife=params["idio_halflife"])

# Build constraint set
cs = (ConstraintSet()
      .add_exclusions(params["exclusions"], bmk_weights)
      .add_grouping_constraint(sector_region['sector'].reindex(bmk_weights.index),
                               bm_weights=bmk_weights, tolerance=params["sector_tolerance"])
      .add_grouping_constraint(sector_region['industry'].reindex(bmk_weights.index),
                               bm_weights=bmk_weights, tolerance=params["industry_tolerance"])
      .add_active_weight_constraint(bmk_weights, bounds=params["stock_active_weight"])
      .add_max_weight_multiple_constraint(bmk_weights, multiple=params["stock_active_weight_multiple"])
      .add_effective_stocks_constraint(bounds=params["n_effective_stocks"])
      .add_tracking_error_constraint(max_te=params["max_te"], risk_model=rm, bm_weights=bmk_weights)
      .add_turnover_constraint(max_turnover=params["max_turnover"], relative_weights=bmk_weights))

# Factor-tilt portfolio: maximise momentum exposure within a 5% annualised TE budget
opt = MaxFactorTiltOptimizer(risk_model=rm, bm_weights=bmk_weights, factor_name='MOM_Momentum',
                             sector_neutral=True)
t0 = time.perf_counter()
result = opt.optimize(long_only=True, constraints=cs)
if hasattr(result, 'nit'):
    nit = result.nit
elif result.solver_stats is not None and result.solver_stats.num_iters is not None:
    nit = result.solver_stats.num_iters
else:
    nit = 'N/A'
print(f"Optimisation completed in {time.perf_counter() - t0:.2f}s | {nit} iterations")

# Output metrics and exposure charts
port_metrics = opt.get_performance_metrics()
bm_metrics = PortfolioMetrics(rm, bmk_weights.values, bmk_weights).get_performance_metrics()

fmt = lambda v: f'{v:.4f}' if isinstance(v, float) else str(v)
metrics_table = pd.DataFrame({
    'Portfolio': {k: fmt(v) for k, v in port_metrics.items()},
    'Benchmark': {k: fmt(v) for k, v in bm_metrics.items()},
})
metrics_table.index.name = 'Metric'
print(metrics_table.to_string())

ExposureCharts(opt.weights, bmk_weights, rm, sector_region).plot_all()