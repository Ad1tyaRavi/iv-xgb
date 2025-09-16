import numpy as np, pandas as pd
from .features import black_scholes_price

def realistic_iv_signal_backtest(df: pd.DataFrame, proba_col: str, prob_threshold: float = 0.5, bid_ask_spread_pct: float = 0.01, group_by_col: str = None) -> dict:
    
    def calculate_metrics(sub_df):
        sig = (sub_df[proba_col] >= prob_threshold).astype(int)
        
        # --- Option parameters ---
        time_to_maturity_days = 30
        risk_free_rate = 0.02

        # --- Simulate P&L ---
        S_t = sub_df['close']
        K_t = sub_df['close']
        iv_t = sub_df['iv']
        t_t = time_to_maturity_days / 365.0

        S_t1 = sub_df['close'].shift(-1)
        iv_t1 = sub_df['iv'].shift(-1)
        t_t1 = (time_to_maturity_days - 1) / 365.0

        entry_price = black_scholes_price('c', S_t, K_t, t_t, risk_free_rate, iv_t) * (1 + bid_ask_spread_pct / 2)
        exit_price = black_scholes_price('c', S_t1, K_t, t_t1, risk_free_rate, iv_t1) * (1 - bid_ask_spread_pct / 2)
        
        ret = exit_price / entry_price - 1.0
        trade_ret = (ret * sig).fillna(0.0)
        
        # --- Metrics ---
        ntrades = int(sig.sum())
        wins = int(((ret > 0) & (sig==1)).sum())
        hitrate = wins / ntrades if ntrades>0 else 0.0
        avg = trade_ret[sig==1].mean() if ntrades>0 else 0.0
        sharpe = trade_ret.mean() / (trade_ret.std() + 1e-9) * np.sqrt(252) if trade_ret.std()>0 else 0.0
        
        return {
            'n_trades': ntrades,
            'win_rate': hitrate,
            'avg_return': float(avg),
            'sharpe_proxy': float(sharpe),
        }

    if group_by_col:
        results = {}
        for group_name, sub_df in df.groupby(group_by_col):
            results[group_name] = calculate_metrics(sub_df)
        return results
    else:
        return calculate_metrics(df)