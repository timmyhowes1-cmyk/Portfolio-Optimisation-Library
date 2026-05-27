import logging
import warnings
from src import load_sp500_prices, get_ff_shares, load_factor_returns, get_sector_region

log = logging.getLogger(__name__)


def load_data():
    factor_returns = load_factor_returns().fillna(0)
    prices = (load_sp500_prices(start_date=factor_returns.index[0],
                                end_date=factor_returns.index[-1])
              .dropna(axis=1, how='all')
              .ffill())

    stock_returns, factor_returns = prices.pct_change().dropna(axis=0, how='all').align(
        factor_returns, join='inner', axis=0
    )

    sector_region = get_sector_region(tickers=stock_returns.columns.tolist())
    ff_mcap = get_ff_shares(tickers=stock_returns.columns.tolist()) * prices.iloc[-1]

    missing = ff_mcap.isna().sum()
    if missing > 0:
        warnings.warn(
            f"{missing} tickers have missing float share data and will receive zero benchmark weight.",
            UserWarning
        )

    bmk_weights = (ff_mcap / ff_mcap.sum()).fillna(0)

    return stock_returns, factor_returns, bmk_weights, sector_region