from abc import ABC, abstractmethod
from typing import Optional, Union
from src import EquityRiskModel
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

    def optimize(self, constraints: Optional[Union[list, 'ConstraintSet']] = None, long_only: bool = True, max_weight: float = 1.0, objective_args: tuple = ()):
        from src.optimizers.constraints import ConstraintSet
        w0 = np.ones(self.n_assets) / self.n_assets

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

    def portfolio_tracking_error(self, w):
        if self._bm_w is not None:
            return self.portfolio_volatility(w - self._bm_w)
        return None

    def portfolio_turnover(self, w, relative_weights=None):
        if relative_weights is None:
            print("Benchmark weights used for turnover calculation...")
            relative_weights = self._bm_w
        return float(np.sum(np.abs(w - relative_weights))) / 2

    def get_holdings(self, floor: float = 0.0) -> pd.DataFrame:
        if self.weights is None:
            raise ValueError("Must run optimize() first")
        df = pd.DataFrame({
            'weight': self.weights,
            'bm_weight': self._bm_w if self._bm_w is not None else np.nan,
        }, index=self.risk_model.tickers)
        df['active_weight'] = df['weight'] - df['bm_weight']
        return df[df['weight'] > floor].sort_values('weight', ascending=False)

    def get_performance_metrics(self, floor:float=0.001):
        if self.weights is None:
            raise ValueError("Must run optimize() first")

        w = self.weights
        ret = self.portfolio_return(w)
        vol = self.portfolio_volatility(w)

        return {
            'expected_return': ret,
            'volatility': vol,
            'tracking_error': self.portfolio_tracking_error(w),
            'sharpe_ratio': ret / vol,
            'num_positions': int(np.sum(w > floor)),
            'n_eff_stocks': int(1 / np.sum(w ** 2))
        }