import numpy as np, pandas as pd
from .features import black_scholes_price

def realistic_iv_signal_backtest(df: pd.DataFrame, proba_col: str, prob_threshold: float = 0.5, bid_ask_spread_pct: float = 0.01, group_by_col: str = None) -> dict:
    
    def calculate_metrics(sub_df):
        sig = (sub_df[proba_col] >= prob_threshold).astype(int)
        
        hold_days = 3
        
        if 'best_offer' in sub_df.columns and 'best_bid' in sub_df.columns:
            # Use real historical option prices
            entry_price = sub_df['best_offer']
            # Exit 3 days later at the bid (selling to close)
            exit_price = sub_df['best_bid'].shift(-hold_days)
        else:
            # Fallback to synthetic Black-Scholes pricing
            time_to_maturity_days = 30
            
            S_t = sub_df['close']
            K_t = sub_df['close']
            iv_t = sub_df['iv']
            t_t = time_to_maturity_days / 365.0
            r_t = sub_df['risk_free_rate'] if 'risk_free_rate' in sub_df.columns else 0.02

            S_t1 = sub_df['close'].shift(-hold_days)
            iv_t1 = sub_df['iv'].shift(-hold_days)
            t_t1 = (time_to_maturity_days - hold_days) / 365.0
            r_t1 = sub_df['risk_free_rate'].shift(-hold_days) if 'risk_free_rate' in sub_df.columns else 0.02

            entry_price = black_scholes_price('c', S_t, K_t, t_t, r_t, iv_t) * (1 + bid_ask_spread_pct / 2)
            exit_price = black_scholes_price('c', S_t1, K_t, t_t1, r_t1, iv_t1) * (1 - bid_ask_spread_pct / 2)
        
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