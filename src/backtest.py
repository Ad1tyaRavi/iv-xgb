import numpy as np
import pandas as pd

def realistic_iv_signal_backtest(df: pd.DataFrame, proba_col: str, use_dynamic_threshold: bool = False, fixed_threshold: float = 0.5, group_by_col: str = None, lookahead: int = 5, spike_threshold: float = 2.0) -> dict:
    """
    Daily Path-Dependent Backtest Engine.
    Simulates $100k portfolio, 3 concurrent slots.
    Executes ATM Straddles with $0.65/leg commission ($1.30 per trade in, $1.30 out).
    Oracle Take-Profit: Exits immediately if IV hits the Z-Score threshold within lookahead.
    Stratifies results by probability tiers.
    """
    if group_by_col:
        results = {}
        for group_name, sub_df in df.groupby(group_by_col):
            results[str(group_name)] = run_simulation(sub_df, proba_col, use_dynamic_threshold, fixed_threshold, lookahead, spike_threshold)
        return results
    else:
        return run_simulation(df, proba_col, use_dynamic_threshold, fixed_threshold, lookahead, spike_threshold)

def run_simulation(df: pd.DataFrame, proba_col: str, use_dynamic_threshold: bool, fixed_threshold: float, lookahead: int = 5, spike_threshold: float = 2.0) -> dict:
    df = df.copy().sort_values('date').reset_index(drop=True)
    
    # Trading logic parameters
    commission_per_leg = 0.65
    commission_per_straddle = commission_per_leg * 2 # $1.30 per straddle
    
    # Portfolio parameters
    initial_capital = 100_000.0
    num_slots = 3
    capital_per_slot = initial_capital / num_slots
    
    active_trades = []
    equity_curve = []
    current_capital = initial_capital
    
    trade_records = []
    
    for i, row in df.iterrows():
        today = row['date']
        
        # 1. Update/Exit Active Trades
        trades_to_keep = []
        for trade in active_trades:
            days_held = trade['days_held'] + 1
            trade['days_held'] = days_held
            
            # Use the same Z-Score logic as labeling for exit
            # We need to re-calculate Z-score or have it in the row
            # For simplicity in this backtest, we use the pre-calculated 'iv_zscore' if available
            if 'iv_zscore' in row:
                iv_target_hit = row['iv_zscore'] >= spike_threshold
            else:
                # Fallback to percentage if zscore not available, but we expect it now
                iv_target_hit = row['iv'] >= trade['entry_iv'] * (1 + 0.15)
            
            time_stop_hit = days_held >= lookahead
            
            if iv_target_hit or time_stop_hit:
                # Exit trade
                exit_price = row['best_bid']
                # P&L calculation: (Exit Price - Entry Price) * Contracts * 100 - Commissions
                gross_pnl = (exit_price - trade['entry_price']) * trade['contracts'] * 100
                net_pnl = gross_pnl - (commission_per_straddle * trade['contracts']) # exit commission
                
                current_capital += trade['allocated_capital'] + net_pnl
                
                trade['exit_date'] = today
                trade['exit_price'] = exit_price
                trade['net_pnl'] = net_pnl
                trade['return_pct'] = net_pnl / trade['allocated_capital']
                trade['exit_reason'] = 'take_profit' if iv_target_hit else 'time_stop'
                
                trade_records.append(trade)
            else:
                trades_to_keep.append(trade)
                
        active_trades = trades_to_keep
        
        # 2. Check for New Entry
        thr = row['optimal_threshold'] if use_dynamic_threshold and 'optimal_threshold' in row else fixed_threshold
        prob = row[proba_col]
        
        if prob >= thr and not np.isnan(row['best_offer']) and row['best_offer'] > 0:
            if len(active_trades) < num_slots:
                # Enter trade
                entry_price = row['best_offer']
                # How many contracts can we buy with capital_per_slot?
                # Cost per contract = entry_price * 100 + commission
                cost_per_contract = (entry_price * 100) + commission_per_straddle
                contracts = int(capital_per_slot // cost_per_contract)
                
                if contracts > 0:
                    allocated_capital = contracts * cost_per_contract
                    current_capital -= allocated_capital
                    
                    active_trades.append({
                        'entry_date': today,
                        'entry_price': entry_price,
                        'entry_iv': row['iv'],
                        'contracts': contracts,
                        'allocated_capital': allocated_capital,
                        'prob': prob,
                        'days_held': 0
                    })
                    
        # 3. Mark to Market Equity
        mtm_capital = current_capital
        for trade in active_trades:
            # Current value of open positions using today's bid
            open_value = (row['best_bid'] * trade['contracts'] * 100) - (commission_per_straddle * trade['contracts'])
            mtm_capital += max(0, open_value) # rough estimate, avoiding negative open value
            
        equity_curve.append({
            'date': today,
            'equity': mtm_capital
        })

    # End of simulation: close remaining trades
    for trade in active_trades:
        trade['exit_date'] = df.iloc[-1]['date']
        trade['net_pnl'] = 0.0 # simplified: mark flat or use last close
        trade['return_pct'] = 0.0
        trade['exit_reason'] = 'end_of_data'
        trade_records.append(trade)

    eq_df = pd.DataFrame(equity_curve)
    if len(eq_df) > 0:
        eq_df['ret'] = eq_df['equity'].pct_change().fillna(0.0)
        daily_vol = eq_df['ret'].std()
        sharpe = (eq_df['ret'].mean() / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0.0
        total_return = (eq_df['equity'].iloc[-1] / initial_capital) - 1.0
        max_drawdown = (eq_df['equity'] / eq_df['equity'].cummax() - 1.0).min()
    else:
        sharpe = 0.0
        total_return = 0.0
        max_drawdown = 0.0

    tr_df = pd.DataFrame(trade_records)
    
    # Probability Tiers
    tiers = {}
    if len(tr_df) > 0:
        bins = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]
        tr_df['tier'] = pd.cut(tr_df['prob'], bins=bins)
        
        for tier, group in tr_df.groupby('tier', observed=True):
            if len(group) > 0:
                tiers[str(tier)] = {
                    'n_trades': int(len(group)),
                    'win_rate': float((group['net_pnl'] > 0).mean()),
                    'avg_pnl': float(group['net_pnl'].mean()),
                    'avg_return': float(group['return_pct'].mean())
                }

    return {
        'n_trades': len(tr_df),
        'win_rate': float((tr_df['net_pnl'] > 0).mean()) if len(tr_df) > 0 else 0.0,
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'probability_tiers': tiers
    }
