import numpy as np, pandas as pd
from scipy.stats import norm

def black_scholes_greeks(flag, S, K, t, r, sigma):
    """
    Calculates Black-Scholes greeks (delta, gamma, vega, theta, vanna, charm)
    """
    # Avoid division by zero
    t = max(t, 1e-6)
    sigma = max(sigma, 1e-6)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    pdf_d1 = norm.pdf(d1)
    
    if flag == 'c':
        delta = norm.cdf(d1)
        gamma = pdf_d1 / (S * sigma * np.sqrt(t))
        vega = S * pdf_d1 * np.sqrt(t)
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(t)) - r * K * np.exp(-r * t) * norm.cdf(d2))
        # Vanna: dDelta / dSigma
        vanna = pdf_d1 * (-d2 / sigma)
        # Charm: dDelta / dTime
        charm = -pdf_d1 * (r / (sigma * np.sqrt(t)) - d2 / (2 * t)) # Simplified version
    elif flag == 'p':
        delta = norm.cdf(d1) - 1
        gamma = pdf_d1 / (S * sigma * np.sqrt(t))
        vega = S * pdf_d1 * np.sqrt(t)
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(t)) + r * K * np.exp(-r * t) * norm.cdf(-d2))
        vanna = pdf_d1 * (-d2 / sigma)
        charm = -pdf_d1 * (r / (sigma * np.sqrt(t)) - d2 / (2 * t))
        
    return delta, gamma, vega, theta, vanna, charm

def black_scholes_price(flag, S, K, t, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if flag == 'c':
        price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    elif flag == 'p':
        price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.ffill()

def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)

def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    log_hl = np.log(high / low)
    return np.sqrt((1 / (4 * window * np.log(2))) * pd.Series(log_hl**2).rolling(window).sum()) * np.sqrt(252)

def rogers_satchell_vol(high: pd.Series, low: pd.Series, open_: pd.Series, close: pd.Series, window: int) -> pd.Series:
    log_ho = np.log(high / open_)
    log_hc = np.log(high / close)
    log_lo = np.log(low / open_)
    log_lc = np.log(low / close)
    rs_squared = log_hc * log_ho + log_lc * log_lo
    return np.sqrt(rs_squared.rolling(window).mean()) * np.sqrt(252)

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
    # Handle division by zero when volume is completely flat (e.g. SPX index)
    out['volume_ratio'] = out['volume'] / out['volume_mean20'].replace(0, np.nan)
    out['volume_ratio'] = out['volume_ratio'].fillna(1.0)
    for w in [5,10,20,30]:
        out[f'realized_vol_{w}'] = realized_vol(out['returns'], w)
        out[f'parkinson_vol_{w}'] = parkinson_vol(out['high'], out['low'], w)
        out[f'rogers_satchell_vol_{w}'] = rogers_satchell_vol(out['high'], out['low'], out['open'], out['close'], w)

    # Surface Velocity (Changes)
    if 'skew_25d' in out.columns:
        out['skew_25d_chg3'] = out['skew_25d'].diff(3)
        out['skew_25d_chg5'] = out['skew_25d'].diff(5)
    if 'skew_10d' in out.columns:
        out['skew_10d_chg3'] = out['skew_10d'].diff(3)
        out['skew_10d_chg5'] = out['skew_10d'].diff(5)
    if 'term_structure_30_10' in out.columns:
        out['term_chg3'] = out['term_structure_30_10'].diff(3)
        out['term_chg5'] = out['term_structure_30_10'].diff(5)

    # Use actual 30D historical volatility as main proxy if available, else fallback
    if 'hist_vol_30' in out.columns:
        out['vix_proxy'] = out['hist_vol_30'] * 100
    else:
        out['vix_proxy'] = out['rogers_satchell_vol_20'] * 100 # Use Rogers-Satchell as the main proxy
    
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
    # --- IV Synthesis ---
    noise = rng.normal(0, 1, size=len(out))
    # Use Rogers-Satchell vol or hist_vol_30 as the base for IV synthesis
    base_vol = out['hist_vol_30'] if 'hist_vol_30' in out.columns else out['rogers_satchell_vol_20']
    out['iv'] = base_vol * (1.2 + 0.1 * noise)
    big_move = out['returns'].abs() > 0.02
    bear = out['market_trend'] == -1
    out.loc[big_move, 'iv'] *= 1.15
    out.loc[bear, 'iv'] *= 1.20
    bull = out['market_trend'] == 1
    out.loc[bull, 'iv'] *= 0.95

    # --- Black-Scholes Greeks Synthesis ---
    time_to_maturity = 30 / 365.0
    risk_free_rate = 0.02
    
    S = out['close'].values
    K = out['close'].values
    t = time_to_maturity
    r = risk_free_rate
    sigma = out['iv'].values
    flag = 'c'

    delta, gamma, vega, theta, vanna, charm = black_scholes_greeks(flag, S, K, t, r, sigma)
    
    out['delta'] = delta
    out['gamma'] = gamma
    out['vega'] = vega
    out['theta'] = theta
    out['vanna'] = vanna
    out['charm'] = charm

    # --- Relative IV & Percentile ---
    rv_base = out['hist_vol_30'] if 'hist_vol_30' in out.columns else out['rogers_satchell_vol_20']
    out['iv_vs_rv'] = out['iv'] / rv_base - 1
    out['iv_percentile_60'] = out['iv'].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x)==60 else np.nan, raw=False
    )
    return out

def finalize_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finalizes the feature table by selecting only a strict whitelist of safe features.
    This prevents accidental data leakage from future-looking columns or identifiers.
    """
    out = df.copy()
    
    # Ensure relative IV metrics are calculated even if not using synthetic greeks
    if 'iv' in out.columns:
        rv_base = out['hist_vol_30'] if 'hist_vol_30' in out.columns else out['rogers_satchell_vol_20']
        if 'iv_vs_rv' not in out.columns:
            out['iv_vs_rv'] = out['iv'] / rv_base - 1
        if 'iv_percentile_60' not in out.columns:
            out['iv_percentile_60'] = out['iv'].rolling(60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x)==60 else np.nan, raw=False
            )

    whitelist = [
        # Identifiers & Targets (passed through but handled in main.py)
        'date', 'iv_spike_3d', 'iv_change_3d', 'max_future_z',
        'best_bid', 'best_offer', 'iv', 'market_trend',
        
        # Underlying & Price Action
        'open', 'high', 'low', 'close', 'volume', 'returns', 'log_returns',
        'high_low_range', 'sma10', 'sma20', 'price_vs_sma10', 'price_vs_sma20',
        'rsi14', 'volume_mean20', 'volume_ratio',
        
        # Volatility & Greeks
        'delta', 'gamma', 'vega', 'theta', 'vanna', 'charm', 'iv_zscore',
        'hist_vol_30', 'risk_free_rate', 'iv_vs_rv', 'iv_percentile_60',
        'vix_proxy', 'vix_proxy_chg', 'skew_25d', 'term_structure_30_10',
        'skew_10d', 'skew_25d_chg3', 'skew_25d_chg5', 'skew_10d_chg3', 'skew_10d_chg5', 
        'term_chg3', 'term_chg5',
    ]
    
    # Add rolling windows of realized volatility
    for w in [5, 10, 20, 30]:
        whitelist.extend([f'realized_vol_{w}', f'parkinson_vol_{w}', f'rogers_satchell_vol_{w}'])
        
    # Only keep columns that are both in the whitelist and in the dataframe
    cols_to_keep = [c for c in whitelist if c in df.columns]
    
    return df[cols_to_keep]
