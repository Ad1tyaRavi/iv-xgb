import numpy as np, pandas as pd

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method='bfill')

def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)

def add_underlying_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['returns'] = out['close'].pct_change()
    out['log_returns'] = np.log(out['close'] / out['close'].shift(1))
    out['high_low_range'] = (out['high'] - out['low']) / out['close']
    out['sma10'] = out['close'].rolling(10).mean()
    out['sma20'] = out['close'].rolling(20).mean()
    out['price_vs_sma10'] = (out['close'] - out['sma10']) / out['sma10']
    out['price_vs_sma20'] = (out['close'] - out['sma20']) / out['sma20']
    out['rsi14'] = rsi(out['close'], 14)
    out['volume_mean20'] = out['volume'].rolling(20).mean()
    out['volume_ratio'] = out['volume'] / out['volume_mean20']
    for w in [5,10,20,30]:
        out[f'realized_vol_{w}'] = realized_vol(out['returns'], w)
    out['vix_proxy'] = out['realized_vol_30'] * 100
    out['vix_proxy_chg'] = out['vix_proxy'].pct_change()
    # Regime
    conds = [
        out['price_vs_sma20'] > 0.02,
        out['price_vs_sma20'] < -0.02
    ]
    out['market_trend'] = np.select(conds, [1,-1], default=0)
    return out

def synthesize_greeks(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    # Base IV from 20d RV
    noise = rng.normal(0, 1, size=len(out))
    out['iv'] = out['realized_vol_20'] * (1.2 + 0.1 * noise)
    # Boost IV during big down moves / bear regimes
    big_move = out['returns'].abs() > 0.02
    bear = out['market_trend'] == -1
    out.loc[big_move, 'iv'] *= 1.15
    out.loc[bear, 'iv'] *= 1.20
    bull = out['market_trend'] == 1
    out.loc[bull, 'iv'] *= 0.95
    # Greeks proxies
    pv10 = out['price_vs_sma10'].fillna(0)
    out['delta'] = 0.5 + 0.2 * np.tanh(pv10 * 2)
    out['gamma'] = 0.1 * np.exp(-2 * (pv10 ** 2))
    out['vega'] = out['iv'] * out['gamma'] * 0.1
    out['theta'] = -out['gamma'] * out['iv'] * out['close'] / 365.0
    # Relative IV
    rv20 = out['realized_vol_20']
    out['iv_vs_rv'] = out['iv'] / rv20 - 1
    # 60d IV percentile
    out['iv_percentile_60'] = out['iv'].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x)==60 else np.nan, raw=False
    )
    return out

def finalize_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    # The original implementation selected a subset of columns, which removed the label columns.
    # We pass the full dataframe through, as the column selection is handled in main.py.
    return df
