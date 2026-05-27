import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import pandas_datareader as pdr

log = logging.getLogger(__name__)


def _load_or_fetch(cache_file: Path, fetch_fn, max_age_days: int = 5):
    if cache_file.exists():
        age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if age.days < max_age_days:
            log.info("Loading %s from cache", cache_file.stem)
            return pd.read_pickle(cache_file)
    result = fetch_fn()
    cache_file.parent.mkdir(exist_ok=True)
    result.to_pickle(cache_file)
    return result


def load_factor_returns(use_cache=True, cache_file=Path(__file__).parent / 'factor_returns.pkl',
                        start_date: datetime = datetime.now() - timedelta(days=3 * 365),
                        end_date: datetime = datetime.now()):
    def fetch():
        ff5 = pdr.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench',
                             start=start_date, end=end_date)[0]
        mom = pdr.DataReader('F-F_Momentum_Factor_daily', 'famafrench',
                             start=start_date, end=end_date)[0]
        factors = ff5.join(mom)
        factors.columns = ['Market', 'SMB_Size', 'HML_Value', 'RMW_Quality',
                           'CMA_Investment', 'RF', 'MOM_Momentum']
        factors['Market-RF'] = factors['Market'] - factors['RF']
        factors = factors.drop(columns=['Market', 'RF'])
        factors /= 100
        return factors

    if use_cache:
        return _load_or_fetch(cache_file, fetch)
    return fetch()


def load_sp500_prices(use_cache=True, cache_file=Path(__file__).parent / 'sp500_prices.pkl',
                      url='https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                      start_date: datetime = datetime.now() - timedelta(days=3 * 365),
                      end_date: datetime = datetime.now()):
    def fetch():
        log.info("Downloading fresh price data")
        tickers = pd.read_html(url, storage_options={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })[0]['Symbol'].tolist()
        return download_prices(tickers, start_date=start_date, end_date=end_date)

    if use_cache:
        return _load_or_fetch(cache_file, fetch)
    return fetch()


def get_ff_shares(tickers=None, use_cache=True,
                  cache_file=Path(__file__).parent / 'sp500_ffmcap.pkl'):
    def fetch():
        ff_shares = {}
        for ticker in tickers:
            try:
                ff_shares[ticker] = yf.Ticker(ticker).info.get('floatShares', None)
            except Exception as e:
                log.warning("Error fetching float shares for %s: %s", ticker, e)
                ff_shares[ticker] = None
        return pd.Series(ff_shares)

    if use_cache:
        return _load_or_fetch(cache_file, fetch)
    return fetch()


def get_sector_region(tickers=None, use_cache=True,
                      cache_file=Path(__file__).parent / 'sp500_sector_region.pkl'):
    def fetch():
        data = []
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                data.append({
                    'ticker':   ticker,
                    'sector':   info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'country':  info.get('country', 'N/A'),
                })
            except Exception as e:
                log.warning("Error fetching sector/region for %s: %s", ticker, e)
                data.append({'ticker': ticker, 'sector': None,
                             'industry': None, 'country': None})
        return pd.DataFrame(data).set_index('ticker')

    if use_cache:
        return _load_or_fetch(cache_file, fetch)
    return fetch()


def download_prices(tickers: list, start_date: datetime, end_date: datetime):
    return yf.download(tickers=tickers, start=start_date, end=end_date,
                       auto_adjust=True, progress=True)['Close']