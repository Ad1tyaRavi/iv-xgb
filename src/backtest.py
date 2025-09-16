import numpy as np, pandas as pd

def simple_iv_signal_backtest(df: pd.DataFrame, proba_col: str, prob_threshold: float = 0.5) -> dict:
    sig = (df[proba_col] >= prob_threshold).astype(int)
    # Return proxy: iv[t+1]/iv[t]-1. Shift to align signal at t with next-day return.
    ret = df['iv'].shift(-1) / df['iv'] - 1.0
    trade_ret = (ret * sig).fillna(0.0)
    ntrades = int(sig.sum())
    wins = int(((ret > 0) & (sig==1)).sum())
    hitrate = wins / ntrades if ntrades>0 else 0.0
    avg = trade_ret[sig==1].mean() if ntrades>0 else 0.0
    # Daily Sharpe proxy (if every day is a decision day, this is loose)
    sharpe = trade_ret.mean() / (trade_ret.std() + 1e-9) * np.sqrt(252) if trade_ret.std()>0 else 0.0
    return {
        'n_trades': ntrades,
        'win_rate': hitrate,
        'avg_return': float(avg),
        'sharpe_proxy': float(sharpe),
    }
