import warnings
import numpy as np
import pandas as pd


class EquityRiskModel:
    def __init__(self, tickers: list, sector_region: pd.DataFrame = None,
                 stock_covariance: pd.DataFrame = None, factor_covariance: pd.DataFrame = None,
                 factor_exposure: pd.DataFrame = None):
        self.tickers = tickers
        self.sector_region = sector_region
        self.stock_covariance = stock_covariance
        self.factor_covariance = factor_covariance
        self.factor_exposure = factor_exposure
        self.idio_risk = None
        self.expected_returns = None
        self.factor_residuals = None
        self._factor_betas_cache = None

        if stock_covariance is not None:
            self._validate_stock_covariance(stock_covariance)

    def _validate_stock_covariance(self, stock_covariance: pd.DataFrame):
        if not np.array_equal(stock_covariance.index.tolist(), self.tickers):
            raise ValueError("Stock covariance tickers don't match.")
        if not stock_covariance.index.equals(stock_covariance.columns):
            raise ValueError("Stock covariance must be square with matching index/columns.")

    def _validate_stock_returns(self, stock_returns: pd.DataFrame) -> None:
        if not np.array_equal(stock_returns.columns.tolist(), self.tickers):
            raise ValueError("Stock returns tickers don't match.")

    def add_stock_covariance(self, stock_returns: pd.DataFrame, annualise: bool = True,
                             halflife: int = None):
        self._validate_stock_returns(stock_returns)
        if halflife is not None:
            ewm = stock_returns.ewm(halflife=halflife)
            last_date = stock_returns.index[-1]
            self.stock_covariance = ewm.cov().xs(last_date, level=0)
            self.expected_returns = ewm.mean().iloc[-1]
        else:
            self.stock_covariance = stock_returns.cov()
            self.expected_returns = self._calculate_expected_returns(stock_returns, annualise=False)

        if annualise:
            self.stock_covariance *= 252
            self.expected_returns *= 252

        return self.stock_covariance

    def add_factor_covariance(self, factor_returns: pd.DataFrame, annualise: bool = True,
                              halflife: int = None):
        if halflife is not None:
            last_date = factor_returns.index[-1]
            self.factor_covariance = factor_returns.ewm(halflife=halflife).cov().xs(last_date, level=0)
        else:
            self.factor_covariance = factor_returns.cov()
        if annualise:
            self.factor_covariance *= 252
        return self.factor_covariance

    @staticmethod
    def _ewma_weights(T: int, halflife: int) -> np.ndarray:
        alpha = 1 - np.exp(-np.log(2) / halflife)
        w = (1 - alpha) ** np.arange(T - 1, -1, -1)  # oldest → newest
        return w / w.sum()

    def add_factor_exposure(self, stock_returns: pd.DataFrame, factor_returns: pd.DataFrame,
                            min_observations: int = 60, halflife: int = None,
                            idio_halflife: int = None):
        self._validate_stock_returns(stock_returns)

        stock_returns, factor_returns = stock_returns.align(factor_returns, join='inner', axis=0)

        T = len(factor_returns)
        n_factors = factor_returns.shape[1]
        min_required = max(min_observations, n_factors + 2)
        factor_names = factor_returns.columns.tolist()
        tickers = stock_returns.columns.tolist()

        X = np.column_stack([np.ones(T), factor_returns.values])  # (T, K+1)
        Y = stock_returns.values                                   # (T, N)

        ew = self._ewma_weights(T, halflife) if halflife is not None else None

        row_names = ['alpha', 'r_squared', 'n_obs'] + factor_names
        result = np.full((len(row_names), len(tickers)), np.nan)

        has_nan = np.isnan(Y).any(axis=0)
        full_idx = np.where(~has_nan)[0]
        nan_idx  = np.where(has_nan)[0]

        def _wls(x, y, w=None):
            # WLS via row-scaling: minimises Σ w_t(y_t - ŷ_t)²
            if w is not None:
                sw = np.sqrt(w)[:, np.newaxis]
                x, y = x * sw, y * sw
            return np.linalg.lstsq(x, y, rcond=None)[0]

        def _r2(y_s, y_hat, w=None):
            y_mean = w @ y_s if w is not None else y_s.mean(axis=0)
            if w is not None:
                ww = w[:, np.newaxis] if y_s.ndim > 1 else w
                ss_res = (ww * (y_s - y_hat) ** 2).sum(axis=0)
                ss_tot = (ww * (y_s - y_mean) ** 2).sum(axis=0)
            else:
                ss_res = np.sum((y_s - y_hat) ** 2, axis=0)
                ss_tot = np.sum((y_s - y_mean) ** 2, axis=0)
            return np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)

        if len(full_idx) > 0:
            y_full = Y[:, full_idx]
            betas  = _wls(X, y_full, ew)
            y_hat  = X @ betas
            r2     = _r2(y_full, y_hat, ew)

            result[0, full_idx] = betas[0]
            result[1, full_idx] = r2
            result[2, full_idx] = T
            result[3:, full_idx] = betas[1:]

        for i in nan_idx:
            y     = Y[:, i]
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

            w_valid = None
            if ew is not None:
                w_valid = ew[valid]
                w_valid = w_valid / w_valid.sum()

            x_v, y_v = X[valid], y[valid]
            betas    = _wls(x_v, y_v[:, np.newaxis], w_valid).ravel()
            y_hat    = x_v @ betas
            r2       = float(_r2(y_v, y_hat, w_valid))

            result[0, i] = betas[0]
            result[1, i] = r2
            result[2, i] = n_obs
            result[3:, i] = betas[1:]

        self.factor_exposure = pd.DataFrame(result, index=row_names, columns=tickers)
        self._factor_betas_cache = None

        # Residuals: ε = Y - Xβ, used for idiosyncratic risk estimation
        full_betas = result[np.r_[0, 3:len(result)], :]  # alpha + factor rows, skip r² and n_obs
        self.factor_residuals = pd.DataFrame(
            Y - X @ full_betas, index=stock_returns.index, columns=tickers
        )

        self.idio_risk = self._calculate_idio_risk(idio_halflife=idio_halflife)
        return self.factor_exposure

    def _calculate_idio_risk(self, idio_halflife: int = None) -> pd.Series:
        if self.factor_residuals is not None:
            resid = self.factor_residuals.reindex(columns=self.tickers)
            if idio_halflife is not None:
                idio_var = resid.ewm(halflife=idio_halflife).var().iloc[-1] * 252
            else:
                idio_var = resid.var() * 252
            return pd.Series(
                np.sqrt(np.maximum(idio_var.values, 0)),
                index=self.tickers, name='idio_volatility'
            )

        # Fallback: (1 - R²) × stock_variance — used only if residuals unavailable
        if self.factor_exposure is None or self.stock_covariance is None:
            raise ValueError("Run add_factor_exposure() first.")
        r_squared = self.factor_exposure.loc['r_squared', self.tickers].values
        stock_variance = np.diag(self.stock_covariance.loc[self.tickers, self.tickers].values)
        return pd.Series(
            np.sqrt((1 - r_squared) * stock_variance),
            index=self.tickers, name='idio_volatility'
        )

    def _calculate_expected_returns(self, stock_returns: pd.DataFrame, annualise: bool = True):
        self._validate_stock_returns(stock_returns)
        return stock_returns.mean(axis=0) * 252 if annualise else stock_returns.mean(axis=0)

    def add_idio_risk(self, idio_halflife: int = None) -> pd.Series:
        self.idio_risk = self._calculate_idio_risk(idio_halflife=idio_halflife)
        return self.idio_risk

    def add_expected_returns(self, stock_returns: pd.DataFrame) -> pd.Series:
        self.expected_returns = self._calculate_expected_returns(stock_returns)
        return self.expected_returns

    def summary(self) -> pd.DataFrame:
        if self.factor_exposure is None:
            raise ValueError("Run add_factor_exposure() first.")
        factor_rows = [r for r in self.factor_exposure.index if r not in {'alpha', 'r_squared', 'n_obs'}]
        return pd.DataFrame({
            'mean_r_squared':        self.factor_exposure.loc['r_squared'].mean(),
            'median_r_squared':      self.factor_exposure.loc['r_squared'].median(),
            'mean_alpha':            self.factor_exposure.loc['alpha'].mean(),
            'mean_n_obs':            self.factor_exposure.loc['n_obs'].mean(),
            'num_stocks':            len(self.tickers),
            'num_factors':           len(factor_rows),
            'pct_valid_regressions': (self.factor_exposure.loc['r_squared'].notna().sum()
                                      / len(self.factor_exposure.columns) * 100),
        }, index=[0])

    def get_factor_betas(self) -> pd.DataFrame:
        if self.factor_exposure is None:
            raise ValueError("Run add_factor_exposure() first.")
        if self._factor_betas_cache is None:
            meta = {'alpha', 'r_squared', 'n_obs'}
            self._factor_betas_cache = self.factor_exposure.loc[
                [r for r in self.factor_exposure.index if r not in meta]
            ]
        return self._factor_betas_cache

    def __repr__(self) -> str:
        state = []
        if self.stock_covariance is not None:
            state.append('stock_cov')
        if self.factor_covariance is not None:
            state.append('factor_cov')
        if self.factor_exposure is not None:
            state.append('exposures')
        status = ', '.join(state) if state else 'uninitialised'
        return f"EquityRiskModel({len(self.tickers)} tickers | {status})"