from .base import BaseOptimizer
import numpy as np

class MinVarianceOptimizer(BaseOptimizer):
    def objective(self, w, *args):
        return self.portfolio_variance(w)

class MeanVarianceOptimizer(BaseOptimizer):
    def objective(self, w, *args):
        risk_aversion = args[0] if args else 0.5
        return -self.portfolio_return(w) + risk_aversion * self.portfolio_variance(w)

    def optimize(self, risk_aversion=0.5, constraints=None, long_only=True, max_weight=1.0):
        return super().optimize(
            constraints=constraints,
            long_only=long_only,
            max_weight=max_weight,
            objective_args=(risk_aversion,)
        )

class MaxSharpeOptimizer(BaseOptimizer):
    def objective(self, w, *args):
        risk_free_rate = args[0] if args else 0.0
        return -(self.portfolio_return(w) - risk_free_rate) / self.portfolio_volatility(w)

    def optimize(self, risk_free_rate=0.0, constraints=None, long_only=True, max_weight=1.0):
        return super().optimize(
            constraints=constraints,
            long_only=long_only,
            max_weight=max_weight,
            objective_args=(risk_free_rate,)
        )

class RiskParityOptimizerConvex(BaseOptimizer):
    def objective(self, w, *args):
        # Spinu's convex formulation: w'Σw - (2/N) * Σ log(wᵢ)
        w_safe = np.maximum(w, 1e-10)
        return self.portfolio_variance(w) - (2.0 / self.n_assets) * np.sum(np.log(w_safe))

    def optimize(self, constraints=None, long_only=True, max_weight=1.0, objective_args=()):
        if not long_only:
            raise ValueError("RiskParityOptimizerConvex requires long_only=True: the Spinu "
                             "formulation uses log(w) which is undefined for non-positive weights.")
        return super().optimize(constraints=constraints, long_only=True,
                                max_weight=max_weight, objective_args=objective_args)

class MinTrackingErrorOptimizer(BaseOptimizer):
    def objective(self, w, *args):
        if self._bm_w is None:
            raise ValueError("Benchmark weights needed for tracking error optimisation. None given.")
        return self.portfolio_variance(w - self._bm_w)

class MaxDiversificationOptimizer(BaseOptimizer):
    def __init__(self, risk_model, bm_weights):
        super().__init__(risk_model, bm_weights)
        self._asset_vols = np.sqrt(np.diag(risk_model.stock_covariance.values))  # (N,)

    def objective(self, w, *args):
        return -(w @ self._asset_vols) / self.portfolio_volatility(w)

class MaxFactorTiltOptimizer(BaseOptimizer):
    def __init__(self, risk_model, bm_weights, factor_name):
        super().__init__(risk_model, bm_weights)

        factor_names = risk_model.get_factor_betas().index.tolist()
        if factor_name not in factor_names:
            raise ValueError(f"Factor '{factor_name}' not in model. Available: {factor_names}")
        self._factor_name = factor_name
        self._tilt_betas = self._B[factor_names.index(factor_name)]  # (N,)

    def objective(self, w, *args):
        return -float(self._tilt_betas @ w)

class MinTurnoverOptimizer(BaseOptimizer):
    def objective(self, w, *args):
        relative_weights = args[0] if args else None
        if relative_weights is None:
            raise ValueError("Weights needed for turnover calculation. None given.")
        # Smooth L1 approximation of turnover for SLSQP gradient stability.
        # True |w - r| has a kink at zero; sqrt((w-r)^2 + ε) is differentiable
        # everywhere with negligible error (ε=1e-8 vs typical weight differences).
        diff = w - relative_weights
        return float(np.sum(np.sqrt(diff ** 2 + 1e-8))) / 2

    def optimize(self, relative_weights=None, constraints=None, long_only=True, max_weight=1.0):
        return super().optimize(
            constraints=constraints,
            long_only=long_only,
            max_weight=max_weight,
            objective_args=(relative_weights,)
        )