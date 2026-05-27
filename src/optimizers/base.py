from abc import ABC, abstractmethod
from typing import Optional, Union
from src import EquityRiskModel
from .constraints import ConstraintSet
from scipy.optimize import minimize
import pandas as pd
import numpy as np

class BaseOptimizer(ABC):
    def __init__(self, risk_model:EquityRiskModel, bm_weights:pd.Series):
        self.risk_model = risk_model
        self.bm_weights = bm_weights
        self.weights = None
        self.n_assets = len(risk_model.tickers)

        # Pre-extract numpy arrays so the optimizer hot path has no DataFrame overhead
        self._B      = risk_model.get_factor_betas().values                            # (K, N)
        self._F      = risk_model.factor_covariance.values                             # (K, K)
        self._idio_sq = risk_model.idio_risk.reindex(risk_model.tickers).values ** 2  # (N,)
        self._mu     = risk_model.expected_returns.reindex(risk_model.tickers).values  # (N,)
        self._bm_w   = bm_weights.reindex(risk_model.tickers).values if bm_weights is not None else None  # (N,)

    @abstractmethod
    def objective(self, w, *args):
        pass

    def optimize(self, constraints: Optional[Union[list, ConstraintSet]] = None, long_only: bool = True, max_weight: float = 1.0, objective_args: tuple = ()):
        w0 = np.ones(self.n_assets) / self.n_assets if self._bm_w is None else self._bm_w

        cs_lower, cs_upper = None, None
        if isinstance(constraints, ConstraintSet):
            cs_lower = constraints.lower_bounds
            cs_upper = constraints.upper_bounds
            constraint_list = list(constraints.constraints)
        elif constraints is None:
            constraint_list = []
        elif isinstance(constraints, dict):
            constraint_list = [constraints]
        else:
            constraint_list = list(constraints)

        constraint_list.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        if long_only:
            lb = np.zeros(self.n_assets)
            ub = np.full(self.n_assets, max_weight)
        else:
            lb = np.full(self.n_assets, -max_weight)
            ub = np.full(self.n_assets, max_weight)

        if cs_lower is not None:
            lb = np.maximum(lb, cs_lower)
        if cs_upper is not None:
            ub = np.minimum(ub, cs_upper)

        bounds = tuple(zip(lb.tolist(), ub.tolist()))

        result = minimize(
            self.objective,
            w0,
            args=objective_args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'ftol': 1e-6, 'maxiter': 500}
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        self.weights = result.x
        return result

    def portfolio_variance(self, w):
        b = self._B @ w
        return float(b @ self._F @ b) + float(np.dot(w * w, self._idio_sq))

    def portfolio_return(self, w):
        return float(self._mu @ w)

    def portfolio_volatility(self, w):
        return np.sqrt(self.portfolio_variance(w))

    def _metrics(self):
        from src.performance import PortfolioMetrics
        if self.weights is None:
            raise ValueError("Must run optimize() first")
        return PortfolioMetrics(self.risk_model, self.weights, self.bm_weights)

    def get_holdings(self, floor: float = 0.0) -> pd.DataFrame:
        return self._metrics().get_holdings(floor=floor)

    def get_performance_metrics(self, floor: float = 0.0001) -> dict:
        return self._metrics().get_performance_metrics(floor=floor)