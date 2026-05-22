import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import pandas_datareader as pdr

def load_factor_returns(use_cache=True, cache_file=Path(__file__).parent / 'factor_returns.pkl',
                    start_date: datetime = datetime.now() - timedelta(days=3 * 365),
                    end_date: datetime = datetime.now()
                    ):

    if use_cache and cache_file.exists():
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age.days < 5:
            print("Loading factor returns from cache...")
            return pd.read_pickle(cache_file)

    ff5_factors = pdr.DataReader(
        'F-F_Research_Data_5_Factors_2x3_daily',
        'famafrench',
        start=start_date,
        end=end_date
    )[0]

    momentum = pdr.DataReader(
        'F-F_Momentum_Factor_daily',
        'famafrench',
        start=start_date,
        end=end_date
    )[0]
    factor_returns = ff5_factors.join(momentum)

    # Rename columns and convert to percentages
    factor_returns.columns = ['Market', 'SMB_Size', 'HML_Value', 'RMW_Quality', 'CMA_Investment', 'RF', 'MOM_Momentum']
    factor_returns['Market-RF'] = factor_returns['Market'] - factor_returns['RF']
    factor_returns = factor_returns.drop(columns=['Market', 'RF'])
    factor_returns /= 100

    # Save to cache
    cache_file.parent.mkdir(exist_ok=True)
    factor_returns.to_pickle(cache_file)

    return factor_returns

def load_sp500_prices(use_cache=True, cache_file=Path(__file__).parent / 'sp500_prices.pkl',
                    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                    start_date: datetime = datetime.now() - timedelta(days=3 * 365),
                    end_date: datetime = datetime.now()
                    ):

    if use_cache and cache_file.exists():
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age.days < 5:
            print("Loading S&P prices from cache...")
            return pd.read_pickle(cache_file)

    # Otherwise download fresh data
    print("Downloading fresh price data...")
    tickers = pd.read_html(url, storage_options={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })[0]['Symbol'].tolist()
    prices = download_prices(tickers, start_date=start_date, end_date=end_date)

    # Save to cache
    cache_file.parent.mkdir(exist_ok=True)
    prices.to_pickle(cache_file)

    return prices

def get_ff_shares(tickers=None, use_cache=True, cache_file=Path(__file__).parent / 'sp500_ffmcap.pkl'):
    if use_cache and cache_file.exists():
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age.days < 5:
            print("Loading S&P mcap data from cache...")
            return pd.read_pickle(cache_file)

    ff_shares = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info

            # Get float shares
            ff_shares[ticker] = info.get('floatShares', None)

        except Exception as e:
            print(f"Error for {ticker}: {e}")
            ff_shares[ticker] = None

    # Save to cache
    cache_file.parent.mkdir(exist_ok=True)
    pd.Series(ff_shares).to_pickle(cache_file)

    return pd.Series(ff_shares)

def get_sector_region(tickers=None, use_cache=True, cache_file=Path(__file__).parent / 'sp500_sector_region.pkl'):
    if use_cache and cache_file.exists():
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age.days < 5:
            print("Loading sector/region data from cache...")
            return pd.read_pickle(cache_file)

    data = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            data.append({
                'ticker': ticker,
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'country': info.get('country', 'N/A'),
                'sectorKey': info.get('sectorKey', 'N/A'),
                'industryKey': info.get('industryKey', 'N/A')
            })
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            data.append({
                'ticker': ticker,
                'sector': None,
                'industry': None,
                'country': None,
                'sectorKey': None,
                'industryKey': None
            })

    # Save to cache
    cache_file.parent.mkdir(exist_ok=True)
    df = pd.DataFrame(data).set_index('ticker')
    df.to_pickle(cache_file)

    return df

def download_prices(tickers:list, start_date:datetime, end_date:datetime):
    return yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=True
        )['Close']
