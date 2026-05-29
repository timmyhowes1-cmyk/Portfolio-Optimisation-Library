import numpy as np
import pandas as pd
import cvxpy as cp

from .slsqp_base import BaseOptimizer
from .cvxpy_base import CVXPYOptimizer


# ── Convex optimizers (CVXPY) ─────────────────────────────────────────────────

class MinVarianceOptimizer(CVXPYOptimizer):
    def cvxpy_objective(self, w):
        return cp.Minimize(
            cp.sum_squares(self._F_sqrt_B @ w) + cp.sum_squares(cp.multiply(self._idio, w))
        )


class MeanVarianceOptimizer(CVXPYOptimizer):
    def optimize(self, risk_aversion=0.5, constraints=None, long_only=True, max_weight=1.0, weight_floor=0.0001):
        self._risk_aversion = risk_aversion
        return super().optimize(constraints=constraints, long_only=long_only,
                                max_weight=max_weight, weight_floor=weight_floor)

    def cvxpy_objective(self, w):
        variance = cp.sum_squares(self._F_sqrt_B @ w) + cp.sum_squares(cp.multiply(self._idio, w))
        return cp.Minimize(-self._mu @ w + self._risk_aversion * variance)


class RiskParityOptimizerConvex(CVXPYOptimizer):
    def optimize(self, constraints=None, long_only=True, max_weight=1.0, weight_floor=0.0001):
        if not long_only:
            raise ValueError("RiskParityOptimizerConvex requires long_only=True: "
                             "log(w) is undefined for non-positive weights.")
        return super().optimize(constraints=constraints, long_only=True,
                                max_weight=max_weight, weight_floor=weight_floor)

    def cvxpy_objective(self, w):
        variance = cp.sum_squares(self._F_sqrt_B @ w) + cp.sum_squares(cp.multiply(self._idio, w))
        return cp.Minimize(variance - (2.0 / self.n_assets) * cp.sum(cp.log(w)))


class MinTrackingErrorOptimizer(CVXPYOptimizer):
    def cvxpy_objective(self, w):
        if self._bm_w is None:
            raise ValueError("Benchmark weights needed for tracking error optimisation. None given.")
        active = w - self._bm_w
        return cp.Minimize(
            cp.sum_squares(self._F_sqrt_B @ active) + cp.sum_squares(cp.multiply(self._idio, active))
        )


class MinTurnoverOptimizer(CVXPYOptimizer):
    def optimize(self, relative_weights=None, constraints=None, long_only=True, max_weight=1.0, weight_floor=0.0001):
        if relative_weights is None:
            raise ValueError("Weights needed for turnover calculation. None given.")
        self._relative_weights = np.asarray(relative_weights)
        return super().optimize(constraints=constraints, long_only=long_only,
                                max_weight=max_weight, weight_floor=weight_floor)

    def cvxpy_objective(self, w):
        return cp.Minimize(cp.norm(w - self._relative_weights, 1) / 2)


class MaxFactorTiltOptimizer(CVXPYOptimizer):
    def __init__(self, risk_model, bm_weights, factor_name, sector_neutral=False):
        super().__init__(risk_model, bm_weights)

        factor_names = risk_model.get_factor_betas().index.tolist()
        if factor_name not in factor_names:
            raise ValueError(f"Factor '{factor_name}' not in model. Available: {factor_names}")
        self._factor_name = factor_name
        self._sector_neutral = sector_neutral

        raw_betas = pd.Series(self._B[factor_names.index(factor_name)], index=risk_model.tickers)

        if sector_neutral:
            if risk_model.sector_region is None:
                raise ValueError("sector_region must be set on the risk model to use sector_neutral=True.")
            sectors = risk_model.sector_region['sector'].reindex(risk_model.tickers)
            sector_counts = sectors.groupby(sectors).transform('count')
            valid = sectors.notna() & (sectors != 'N/A') & (sector_counts >= 5)
            means = raw_betas.groupby(sectors).transform('mean')
            stds  = raw_betas.groupby(sectors).transform('std').clip(lower=0.01)
            z = ((raw_betas - means) / stds).clip(-3, 3)
            z[~valid] = 0.0
            self._tilt_betas = z.values
        else:
            self._tilt_betas = raw_betas.values

    def cvxpy_objective(self, w):
        return cp.Maximize(self._tilt_betas @ w)

    def get_performance_metrics(self):
        metrics = super().get_performance_metrics()
        metrics['factor_tilt'] = float(self._tilt_betas @ self.weights)
        return metrics


# ── Non-convex optimizers (SLSQP) ────────────────────────────────────────────

class MaxSharpeOptimizer(BaseOptimizer):
    def objective(self, w, *args):
        risk_free_rate = args[0] if args else 0.0
        return -(self.portfolio_return(w) - risk_free_rate) / self.portfolio_volatility(w)

    def objective_gradient(self, w, *args):
        risk_free_rate = args[0] if args else 0.0
        sigma = self.portfolio_volatility(w)
        excess = self.portfolio_return(w) - risk_free_rate
        grad_sigma = self.portfolio_variance_gradient(w) / (2 * sigma)
        return -(self._mu * sigma - excess * grad_sigma) / sigma ** 2

    def optimize(self, risk_free_rate=0.0, constraints=None, long_only=True, max_weight=1.0, weight_floor=0.0001):
        return super().optimize(constraints=constraints, long_only=long_only, max_weight=max_weight,
                                objective_args=(risk_free_rate,), weight_floor=weight_floor)


class MaxDiversificationOptimizer(BaseOptimizer):
    def __init__(self, risk_model, bm_weights):
        super().__init__(risk_model, bm_weights)
        self._asset_vols = np.sqrt(np.diag(risk_model.stock_covariance.values))  # (N,)

    def objective(self, w, *args):
        return -(w @ self._asset_vols) / self.portfolio_volatility(w)

    def objective_gradient(self, w, *args):
        sigma = self.portfolio_volatility(w)
        weighted_vol = w @ self._asset_vols
        grad_sigma = self.portfolio_variance_gradient(w) / (2 * sigma)
        return -(self._asset_vols * sigma - weighted_vol * grad_sigma) / sigma ** 2