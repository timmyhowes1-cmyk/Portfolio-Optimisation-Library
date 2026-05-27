import warnings
import numpy as np
import pandas as pd
from typing import Union, Optional


class ConstraintSet:
    def __init__(self, constraints: list = None):
        self.constraints = constraints if constraints is not None else []
        self.lower_bounds: Optional[np.ndarray] = None
        self.upper_bounds: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.constraints)

    def __repr__(self):
        return f"ConstraintSet({len(self.constraints)} constraint(s))"

    def add_grouping_constraint(self, grouping: pd.Series, bm_weights: pd.Series = None,
                                tolerance: Union[float, list, np.ndarray] = 0.05):
        group_mask = pd.get_dummies(grouping)
        n_groups = len(group_mask.columns)

        if bm_weights is None and isinstance(tolerance, float) and tolerance * n_groups < 1:
            warnings.warn("Tolerance too small — constraint not added.", UserWarning)
            return self

        if isinstance(tolerance, pd.DataFrame):
            tolerance = tolerance.reindex(group_mask.columns).values
        tol = np.asarray(tolerance)
        if tol.ndim == 0:
            low_tol, high_tol = -float(tol), float(tol)
        elif tol.ndim == 1:
            if len(tol) != 2:
                raise ValueError("1-D tolerance must be [lower, upper].")
            low_tol, high_tol = tol.min(), tol.max()
        else:
            if tol.shape != (n_groups, 2):
                raise ValueError(f"Per-group tolerance must have shape ({n_groups}, 2).")
            low_tol, high_tol = tol.min(axis=1), tol.max(axis=1)

        mask = group_mask.values
        if bm_weights is not None:
            bm = bm_weights.reindex(group_mask.index).fillna(0).values
            lower_bound = bm @ mask + low_tol
            upper_bound = bm @ mask + high_tol
        else:
            lower_bound = low_tol
            upper_bound = high_tol

        self.constraints.extend([
            {'type': 'ineq', 'fun': lambda w, m=mask: upper_bound - m.T @ w},
            {'type': 'ineq', 'fun': lambda w, m=mask: m.T @ w - lower_bound},
        ])
        return self

    def add_attribute_constraint(self, attribute: pd.Series, bounds: Union[tuple, list, np.ndarray],
                                 bm_weights: pd.Series = None):
        index = bm_weights.index if bm_weights is not None else attribute.index
        attr_values = attribute.reindex(index).values
        lower_bound, upper_bound = min(bounds), max(bounds)

        if bm_weights is not None:
            bm_attr = float(bm_weights.values @ attr_values)
            lower_bound = bm_attr + lower_bound
            upper_bound = bm_attr + upper_bound

        self.constraints.extend([
            {'type': 'ineq', 'fun': lambda w, a=attr_values: upper_bound - w @ a},
            {'type': 'ineq', 'fun': lambda w, a=attr_values: w @ a - lower_bound},
        ])
        return self

    def add_quadratic_constraint(self, Q: np.ndarray, bounds: Union[tuple, list, np.ndarray],
                                 c: np.ndarray = None, b: float = 0.0):
        """Constrains lb <= w'Qw + c'w + b <= ub. Pass None in bounds for a one-sided constraint."""
        lower_bound, upper_bound = bounds[0], bounds[1]
        q_arr = np.asarray(Q)
        c_arr = np.zeros(q_arr.shape[0]) if c is None else np.asarray(c)

        if upper_bound is not None:
            self.constraints.append(
                {'type': 'ineq', 'fun': lambda w, q=q_arr, cv=c_arr: upper_bound - (w @ q @ w + cv @ w + b)}
            )
        if lower_bound is not None:
            self.constraints.append(
                {'type': 'ineq', 'fun': lambda w, q=q_arr, cv=c_arr: (w @ q @ w + cv @ w + b) - lower_bound}
            )
        return self

    def add_effective_stocks_constraint(self, bounds: Union[tuple, list, np.ndarray]):
        """Constrains effective number of stocks N_eff = 1/HHI = 1/(w'w).

        Equivalent to add_quadratic_constraint with Q=I, but avoids constructing
        an n×n identity matrix at setup time.
        """
        n_min, n_max = min(bounds), max(bounds)

        # N_eff >= n_min  =>  w'w <= 1 / n_min
        if n_min > 0:
            hhi_upper = 1.0 / n_min
            self.constraints.append({'type': 'ineq', 'fun': lambda w: hhi_upper - w @ w})

        # N_eff <= n_max  =>  w'w >= 1/n_max
        if np.isfinite(n_max):
            hhi_lower = 1.0 / n_max
            self.constraints.append({'type': 'ineq', 'fun': lambda w: w @ w - hhi_lower})
        return self

    def add_max_weight_constraint(self, max_weight: Union[float, pd.Series],
                                  bm_weights: pd.Series = None):
        """Per-stock upper bound on portfolio weight.

        max_weight can be a scalar (uniform cap) or a Series (per-stock caps,
        requires bm_weights for index alignment).
        Returns a vector constraint — one element per stock.
        """
        if isinstance(max_weight, pd.Series):
            if bm_weights is None:
                raise ValueError("bm_weights required for alignment when max_weight is a Series.")
            ub = max_weight.reindex(bm_weights.index).values
        else:
            ub = float(max_weight)

        self.constraints.append({'type': 'ineq', 'fun': lambda w, cap=ub: cap - w})
        return self

    def add_exclusions(self, exclude: list, bm_weights: pd.Series):
        ub = np.where(bm_weights.index.isin(exclude), 0.0, np.inf)
        self.upper_bounds = ub if self.upper_bounds is None else np.minimum(self.upper_bounds, ub)
        return self

    def add_active_weight_constraint(self, bm_weights: pd.Series,
                                     bounds: Union[float, list, np.ndarray]):
        """Per-stock active weight bounds: lb <= w_i - bm_i <= ub.

        Stored as per-stock bounds rather than SLSQP constraints so the solver
        treats them as simple box bounds, which is far more efficient.
        """
        bm_vals = bm_weights.values
        b = np.asarray(bounds)
        if b.ndim == 0:
            lower_bound, upper_bound = -float(b), float(b)
        else:
            lower_bound, upper_bound = float(b.min()), float(b.max())

        new_lower = bm_vals + lower_bound
        new_upper = bm_vals + upper_bound
        self.lower_bounds = new_lower if self.lower_bounds is None else np.maximum(self.lower_bounds, new_lower)
        self.upper_bounds = new_upper if self.upper_bounds is None else np.minimum(self.upper_bounds, new_upper)
        return self

    def add_tracking_error_constraint(self, max_te: float, risk_model, bm_weights: pd.Series):
        """Upper bound on annualised tracking error vs benchmark: TE <= max_te."""
        B = risk_model.get_factor_betas().values
        F = risk_model.factor_covariance.values
        idio_sq = risk_model.idio_risk.reindex(bm_weights.index).values ** 2
        bm_w = bm_weights.values

        def _te_constraint(w):
            active = w - bm_w
            b = B @ active
            return max_te ** 2 - (float(b @ F @ b) + float(np.dot(active * active, idio_sq)))

        self.constraints.append({'type': 'ineq', 'fun': _te_constraint})
        return self

    def add_turnover_constraint(self, max_turnover: float, relative_weights: pd.Series):
        ref = relative_weights.values

        def _turnover_constraint(w):
            diff = w - ref
            return max_turnover - 0.5 * float(np.sum(np.sqrt(diff ** 2 + 1e-8)))

        self.constraints.append({'type': 'ineq', 'fun': _turnover_constraint})
        return self

    def add_max_weight_multiple_constraint(self, bm_weights: pd.Series,
                                           multiple: Union[float, pd.Series]):
        """Per-stock upper bound: w_i <= multiple * bm_i.

        Stored as per-stock bounds rather than SLSQP constraints so the solver
        treats them as simple box bounds, which is far more efficient.
        """
        bm_vals = bm_weights.values
        if isinstance(multiple, pd.Series):
            mult = multiple.reindex(bm_weights.index).values
        else:
            mult = float(multiple)

        new_upper = mult * bm_vals
        self.upper_bounds = new_upper if self.upper_bounds is None else np.minimum(self.upper_bounds, new_upper)
        return self