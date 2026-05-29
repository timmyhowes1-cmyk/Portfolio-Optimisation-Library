import warnings
import numpy as np
import pandas as pd
from typing import Union, Optional


class ConstraintSet:
    def __init__(self):
        self.constraints: list = []
        self.lower_bounds: Optional[np.ndarray] = None
        self.upper_bounds: Optional[np.ndarray] = None
        self._exclusion_mask: Optional[np.ndarray] = None
        self._cvxpy_specs: list = []

    def __len__(self):
        return len(self.constraints)

    def __repr__(self):
        return f"ConstraintSet({len(self.constraints)} constraint(s))"

    def add_grouping_constraint(self, grouping: pd.Series, bm_weights: pd.Series = None,
                                tolerance: Union[float, list, np.ndarray] = 0.05):
        clean = grouping.replace('N/A', np.nan).dropna()
        group_mask = pd.get_dummies(clean).reindex(grouping.index, fill_value=0)
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
        mask_T = mask.T
        if bm_weights is not None:
            bm = bm_weights.reindex(group_mask.index).fillna(0).values
            lower_bound = bm @ mask + low_tol
            upper_bound = bm @ mask + high_tol
        else:
            lower_bound = low_tol
            upper_bound = high_tol

        self.constraints.extend([
            {'type': 'ineq', 'fun': lambda w, m=mask: upper_bound - m.T @ w,   'jac': lambda w, mT=mask_T: -mT},
            {'type': 'ineq', 'fun': lambda w, m=mask: m.T @ w - lower_bound,   'jac': lambda w, mT=mask_T:  mT},
        ])
        self._cvxpy_specs.append({'type': 'grouping', 'mask_T': mask_T, 'lower_bound': lower_bound, 'upper_bound': upper_bound})
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

        neg_attr = -attr_values
        self.constraints.extend([
            {'type': 'ineq', 'fun': lambda w, a=attr_values: upper_bound - w @ a, 'jac': lambda w, na=neg_attr: na},
            {'type': 'ineq', 'fun': lambda w, a=attr_values: w @ a - lower_bound, 'jac': lambda w, a=attr_values: a},
        ])
        self._cvxpy_specs.append({'type': 'attribute', 'attr': attr_values, 'lower_bound': lower_bound, 'upper_bound': upper_bound})
        return self

    def add_quadratic_constraint(self, Q: np.ndarray, bounds: Union[tuple, list, np.ndarray],
                                 c: np.ndarray = None, b: float = 0.0):
        """Constrains lb <= w'Qw + c'w + b <= ub. Pass None in bounds for a one-sided constraint."""
        lower_bound, upper_bound = bounds[0], bounds[1]
        q_arr = np.asarray(Q)
        c_arr = np.zeros(q_arr.shape[0]) if c is None else np.asarray(c)

        if upper_bound is not None:
            self.constraints.append({
                'type': 'ineq',
                'fun': lambda w, q=q_arr, cv=c_arr: upper_bound - (w @ q @ w + cv @ w + b),
                'jac': lambda w, q=q_arr, cv=c_arr: -(2 * q @ w + cv),
            })
        if lower_bound is not None:
            self.constraints.append({
                'type': 'ineq',
                'fun': lambda w, q=q_arr, cv=c_arr: (w @ q @ w + cv @ w + b) - lower_bound,
                'jac': lambda w, q=q_arr, cv=c_arr:  (2 * q @ w + cv),
            })
        self._cvxpy_specs.append({'type': 'quadratic', 'Q': q_arr, 'c': c_arr, 'b': b, 'lower_bound': lower_bound, 'upper_bound': upper_bound})
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
            self.constraints.append({'type': 'ineq', 'fun': lambda w: hhi_upper - w @ w, 'jac': lambda w: -2 * w})

        # N_eff <= n_max  =>  w'w >= 1/n_max
        if np.isfinite(n_max):
            hhi_lower = 1.0 / n_max
            self.constraints.append({'type': 'ineq', 'fun': lambda w: w @ w - hhi_lower, 'jac': lambda w:  2 * w})
        self._cvxpy_specs.append({'type': 'effective_stocks', 'n_min': n_min, 'n_max': n_max})
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

        self.constraints.append({'type': 'ineq', 'fun': lambda w, cap=ub: cap - w, 'jac': lambda w: -np.eye(len(w))})
        self._cvxpy_specs.append({'type': 'max_weight', 'ub': ub})
        return self

    def add_exclusions(self, exclude: list, bm_weights: pd.Series):
        self._exclusion_mask = np.array(bm_weights.index.isin(exclude))
        mask = self._exclusion_mask
        neg_mask = -mask.astype(float)
        self.constraints.append({'type': 'ineq', 'fun': lambda w, m=mask: -np.sum(w[m]), 'jac': lambda w, nm=neg_mask: nm})
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

        def _te_jac(w):
            active = w - bm_w
            return -2 * (B.T @ (F @ (B @ active)) + idio_sq * active)

        self.constraints.append({'type': 'ineq', 'fun': _te_constraint, 'jac': _te_jac})
        self._cvxpy_specs.append({'type': 'tracking_error', 'F_sqrt_B': np.linalg.cholesky(F).T @ B, 'idio': np.sqrt(idio_sq), 'bm_w': bm_w, 'max_te': max_te})
        return self

    def add_turnover_constraint(self, max_turnover: float, relative_weights: pd.Series):
        ref = relative_weights.values

        def _turnover_constraint(w):
            return max_turnover - 0.5 * float(np.sum(np.abs(w - ref)))

        def _turnover_jac(w):
            return -0.5 * np.sign(w - ref)

        self.constraints.append({'type': 'ineq', 'fun': _turnover_constraint, 'jac': _turnover_jac})
        self._cvxpy_specs.append({'type': 'turnover', 'ref': ref, 'max_turnover': max_turnover})
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

    def to_cvxpy(self, w) -> list:
        import cvxpy as cp
        if self.constraints and not self._cvxpy_specs:
            warnings.warn(
                "This ConstraintSet was built with raw SLSQP dicts and has no CVXPY equivalents. "
                "All constraints will be ignored by the CVXPY solver. Use the fluent add_*() methods.",
                UserWarning,
            )
        constraints = []
        for s in self._cvxpy_specs:
            t = s['type']
            if t == 'grouping':
                constraints += [s['mask_T'] @ w <= s['upper_bound'], s['mask_T'] @ w >= s['lower_bound']]
            elif t == 'attribute':
                constraints += [s['attr'] @ w <= s['upper_bound'], s['attr'] @ w >= s['lower_bound']]
            elif t == 'quadratic':
                expr = cp.quad_form(w, s['Q']) + s['c'] @ w + s['b']
                if s['upper_bound'] is not None:
                    constraints.append(expr <= s['upper_bound'])
                if s['lower_bound'] is not None:
                    constraints.append(expr >= s['lower_bound'])
            elif t == 'effective_stocks':
                if s['n_min'] > 0:
                    constraints.append(cp.sum_squares(w) <= 1.0 / s['n_min'])
                if np.isfinite(s['n_max']):
                    import warnings
                    warnings.warn("Effective stocks upper bound (n_max) is non-convex and skipped in CVXPY mode.", UserWarning)
            elif t == 'tracking_error':
                active = w - s['bm_w']
                te_sq = cp.sum_squares(s['F_sqrt_B'] @ active) + cp.sum_squares(cp.multiply(s['idio'], active))
                constraints.append(te_sq <= s['max_te'] ** 2)
            elif t == 'turnover':
                constraints.append(cp.norm(w - s['ref'], 1) <= 2 * s['max_turnover'])
            elif t == 'max_weight':
                constraints.append(w <= s['ub'])
        return constraints