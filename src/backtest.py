import numpy as np, pandas as pd

def realistic_iv_signal_backtest(df: pd.DataFrame, proba_col: str, prob_threshold: float = 0.5, bid_ask_spread_pct: float = 0.01, group_by_col: str = None) -> dict:
    
    if 'trade_ret' not in df.columns:
        raise ValueError("Backtester requires a pre-calculated 'trade_ret' column to avoid chronological gaps.")

    def calculate_metrics(sub_df):
        sig = (sub_df[proba_col] >= prob_threshold).astype(int)
        ret = sub_df['trade_ret']
        
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