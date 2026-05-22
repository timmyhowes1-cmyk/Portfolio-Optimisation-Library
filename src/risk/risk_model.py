import pandas as pd
import warnings
import numpy as np
from src.data.data_loader import *

class EquityRiskModel:
    def __init__(self, tickers:list, sector_region:pd.DataFrame=None,
                 stock_covariance:pd.DataFrame=None, factor_covariance:pd.DataFrame=None,
                 factor_exposure:pd.DataFrame=None):
        self.tickers = tickers
        self.sector_region = sector_region
        self.stock_covariance = stock_covariance
        self.factor_covariance = factor_covariance
        self.factor_exposure = factor_exposure
        self.idio_risk = None
        self.expected_returns = None
        self._factor_betas_cache = None

        if stock_covariance is not None:
            self._validate_stock_covariance(stock_covariance)

    def _validate_stock_covariance(self, stock_covariance:pd.DataFrame):
        if not np.array_equal(stock_covariance.index.tolist(), self.tickers):
            raise ValueError(
                f"Stock covariance tickers don't match. "
            )
        if not stock_covariance.index.equals(stock_covariance.columns):
            raise ValueError("Stock covariance must be square with matching index/columns")

    def _validate_stock_returns(self, stock_returns: pd.DataFrame) -> None:
        if not np.array_equal(stock_returns.columns.tolist(), self.tickers):
            raise ValueError(
                f"Stock returns tickers don't match. "
            )

    def add_stock_covariance(self, stock_returns, annualise:bool=True):
        self._validate_stock_returns(stock_returns)
        self.stock_covariance = stock_returns.cov()
        self.expected_returns = self._calculate_expected_returns(stock_returns, annualise=False)

        if annualise:
            self.stock_covariance *= 252
            self.expected_returns *= 252

        return self.stock_covariance

    def add_factor_covariance(self, factor_returns, annualise:bool=True):
        self.factor_covariance = factor_returns.cov()
        if annualise:
            self.factor_covariance *= 252

        return self.factor_covariance

    def add_factor_exposure(self, stock_returns, factor_returns, min_observations:int=60):
        self._validate_stock_returns(stock_returns)

        stock_returns, factor_returns = stock_returns.align(
            factor_returns, join='inner', axis=0
        )

        n_factors = factor_returns.shape[1]
        min_required = max(min_observations, n_factors + 2)
        factor_names = factor_returns.columns.tolist()
        tickers = stock_returns.columns.tolist()

        # Design matrix with intercept column: (T, K+1)
        X = np.column_stack([np.ones(len(factor_returns)), factor_returns.values])
        Y = stock_returns.values  # (T, N)

        row_names = ['alpha', 'r_squared', 'n_obs'] + factor_names
        result = np.full((len(row_names), len(tickers)), np.nan)

        has_nan = np.isnan(Y).any(axis=0)  # (N,) bool
        full_idx = np.where(~has_nan)[0]
        nan_idx = np.where(has_nan)[0]

        # Batch solve for all stocks with complete data in one lstsq call
        if len(full_idx) > 0:
            Y_full = Y[:, full_idx]                                    # (T, N_full)
            betas, _, _, _ = np.linalg.lstsq(X, Y_full, rcond=None)  # (K+1, N_full)

            Y_hat = X @ betas
            Y_mean = Y_full.mean(axis=0)
            ss_res = np.sum((Y_full - Y_hat) ** 2, axis=0)
            ss_tot = np.sum((Y_full - Y_mean) ** 2, axis=0)
            r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)

            result[0, full_idx] = betas[0]    # alpha
            result[1, full_idx] = r2           # r_squared
            result[2, full_idx] = len(X)       # n_obs
            result[3:, full_idx] = betas[1:]   # factor betas

        # Per-stock numpy lstsq for stocks with missing data
        for i in nan_idx:
            y = Y[:, i]
            valid = ~np.isnan(y)
            n_obs = int(valid.sum())

            if n_obs < min_required:
                warnings.warn(
                    f"Stock {tickers[i]} has only {n_obs} clean observations "
                    f"(minimum: {min_required}). Returning NaN.",
                    UserWarning
                )
                result[2, i] = n_obs
                continue

            X_clean = X[valid]
            y_clean = y[valid]
            betas, _, _, _ = np.linalg.lstsq(X_clean, y_clean, rcond=None)

            y_hat = X_clean @ betas
            ss_res = float(np.sum((y_clean - y_hat) ** 2))
            ss_tot = float(np.sum((y_clean - y_clean.mean()) ** 2))

            result[0, i] = betas[0]
            result[1, i] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            result[2, i] = n_obs
            result[3:, i] = betas[1:]

        self.factor_exposure = pd.DataFrame(result, index=row_names, columns=tickers)
        self._factor_betas_cache = None  # invalidate cache

        if self.stock_covariance is not None:
            self.idio_risk = self._calculate_idio_risk()

        return self.factor_exposure

    def _calculate_idio_risk(self):
        if self.factor_exposure is None or self.stock_covariance is None:
            raise ValueError(
                "Both factor_exposure and stock_covariance must be set. "
                "Run add_factor_exposure() and add_stock_covariance() first."
            )

        common_tickers = self.factor_exposure.columns.intersection(
            self.stock_covariance.index
        )

        if len(common_tickers) != len(self.tickers):
            warnings.warn(
                f"Only {len(common_tickers)}/{len(self.tickers)} tickers aligned "
                f"between factor exposures and covariance matrix"
            )

        r_squared = self.factor_exposure.loc['r_squared', common_tickers].values
        stock_variance = np.diag(self.stock_covariance.loc[common_tickers, common_tickers].values)
        idio_variance = (1 - r_squared) * stock_variance

        return pd.Series(
            np.sqrt(idio_variance),
            index=common_tickers,
            name='idio_volatility'
        )

    def _calculate_expected_returns(self, stock_returns:pd.DataFrame, annualise=True):
        self._validate_stock_returns(stock_returns)
        return stock_returns.mean(axis=0) * 252 if annualise else stock_returns.mean(axis=0)

    def add_idio_risk(self):
        self.idio_risk = self._calculate_idio_risk()
        return self.idio_risk

    def add_expected_returns(self, stock_returns:pd.DataFrame):
        self.expected_returns = self._calculate_expected_returns(stock_returns)
        return self.expected_returns

    def summary(self):
        if self.factor_exposure is None:
            raise ValueError("Factor exposures not calculated. Run add_factor_exposure() first.")

        factor_cols = [
            col for col in self.factor_exposure.index
            if col not in ['alpha', 'r_squared', 'n_obs']
        ]

        return pd.DataFrame({
            'mean_r_squared': self.factor_exposure.loc['r_squared'].mean(),
            'median_r_squared': self.factor_exposure.loc['r_squared'].median(),
            'mean_alpha': self.factor_exposure.loc['alpha'].mean(),
            'mean_n_obs': self.factor_exposure.loc['n_obs'].mean() if 'n_obs' in self.factor_exposure.index else np.nan,
            'num_stocks': len(self.tickers),
            'num_factors': len(factor_cols),
            'pct_valid_regressions': (self.factor_exposure.loc['r_squared'].notna().sum() /
                                      len(self.factor_exposure.columns) * 100)
        }, index=[0])

    def get_factor_betas(self):
        if self.factor_exposure is None:
            raise ValueError("Factor exposures not calculated.")

        if self._factor_betas_cache is None:
            metadata_rows = {'alpha', 'r_squared', 'n_obs'}
            beta_rows = [row for row in self.factor_exposure.index if row not in metadata_rows]
            self._factor_betas_cache = self.factor_exposure.loc[beta_rows]

        return self._factor_betas_cache