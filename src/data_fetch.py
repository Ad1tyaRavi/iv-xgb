
import os
import time
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Handle MultiIndex columns from yfinance gracefully
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in tup if c!='']) for tup in df.columns.values]
    else:
        df.columns = [str(c) for c in df.columns]
    return df

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # Try to locate standard OHLCV columns regardless of case or prefixes
    cols = {c.lower(): c for c in df.columns}
    def pick(names):
        for n in names:
            if n in cols:
                return cols[n]
        return None
    open_col  = pick(['open','open_spy','spy_open','open_adj'])
    high_col  = pick(['high','spy_high'])
    low_col   = pick(['low','spy_low'])
    close_col = pick(['close','spy_close','adj close','adj_close','close_adj'])
    vol_col   = pick(['volume','spy_volume'])
    # Build a normalized frame
    out = pd.DataFrame(index=df.index.copy())
    if open_col:  out['open']  = df[open_col].astype(float)
    if high_col:  out['high']  = df[high_col].astype(float)
    if low_col:   out['low']   = df[low_col].astype(float)
    if close_col: out['close'] = df[close_col].astype(float)
    if vol_col:   out['volume']= df[vol_col].astype(float)
    out = out.reset_index()
    # Ensure there's a 'date' column regardless of index name
    if 'Date' in out.columns: 
        out = out.rename(columns={'Date':'date'})
    if 'date' not in out.columns:
        # last resort: if first column looks like datetime, call it date
        first = out.columns[0]
        if np.issubdtype(out[first].dtype, np.datetime64) or 'date' in first.lower():
            out = out.rename(columns={first: 'date'})
        else:
            out['date'] = pd.to_datetime(out.index)
    out['date'] = pd.to_datetime(out['date'])
    # Return only the standard set
    keep = ['date','open','high','low','close','volume']
    for k in keep:
        if k not in out.columns:
            out[k] = np.nan
    return out[keep]

def _try_yf_download(ticker: str, start: str, end: str|None, attempts: int = 4, sleep_base: float = 1.5) -> pd.DataFrame:
    last_err = None
    for i in range(attempts):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            last_err = e
        # backoff
        time.sleep(sleep_base * (2 ** i) + np.random.random()*0.25)
    if last_err:
        raise last_err
    return pd.DataFrame()

def _try_yf_history(ticker: str, start: str, end: str|None) -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=start, end=end, interval="1d", auto_adjust=False)
        return hist
    except Exception:
        return pd.DataFrame()

def _synthesize_ohlcv(start: str, end: str|None, n: int = 3500, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic but realistic-looking OHLCV series (GBM with noise)."""
    rng = np.random.default_rng(seed)
    if end is None:
        end = dt.date.today().isoformat()
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    dates = pd.bdate_range(start_dt, end_dt)
    if len(dates) == 0:
        dates = pd.bdate_range(end_dt - pd.Timedelta(days=n*1.6), end_dt)
    N = len(dates)
    mu, sigma = 0.08, 0.18  # yearly drift/vol
    dt_year = 1/252
    shocks = rng.normal((mu - 0.5*sigma**2)*dt_year, sigma*np.sqrt(dt_year), size=N)
    price = 100*np.exp(np.cumsum(shocks))
    # OHLC with small intraday ranges
    high = price*(1 + np.abs(rng.normal(0.0015, 0.002, size=N)))
    low  = price*(1 - np.abs(rng.normal(0.0015, 0.002, size=N)))
    open_ = price*(1 + rng.normal(0, 0.001, size=N))
    close = price
    volume = rng.integers(3e7, 1.2e8, size=N)
    df = pd.DataFrame({
        'date': dates, 'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    })
    return df

def fetch_underlying_ohlcv(ticker: str, start: str, end: str|None=None, allow_synthetic: bool = True) -> pd.DataFrame:
    """Robust OHLCV fetch with fallbacks and MultiIndex normalization.
    Order:
      1) yf.download (retry/backoff)
      2) Ticker.history fallback
      3) If still empty (e.g., rate-limited), synthesize OHLCV (if allowed)
    """
    if end is None:
        end = dt.date.today().isoformat()

    # 1) Try yf.download with retries
    try:
        raw = _try_yf_download(ticker, start, end)
        if raw is not None and len(raw) > 0:
            raw = _flatten_columns(raw)
            normalized = _normalize_ohlcv(raw)
            # Check if normalization produced valid data before returning
            if not normalized['close'].isnull().all():
                return normalized
    except Exception:
        pass

    # 2) Fallback to history()
    raw2 = _try_yf_history(ticker, start, end)
    if raw2 is not None and len(raw2) > 0:
        raw2 = _flatten_columns(raw2)
        return _normalize_ohlcv(raw2)

    # 3) Last resort: synthesize
    if allow_synthetic:
        df = _synthesize_ohlcv(start, end)
        return df[['date','open','high','low','close','volume']]

    # If we reach here, fail clearly
    raise RuntimeError("Failed to fetch OHLCV via yfinance (rate limit/empty), and synthetic fallback disabled.")
