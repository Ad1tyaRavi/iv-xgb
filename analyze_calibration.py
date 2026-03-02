import json
import pandas as pd
import os

# Load metrics to get the spike threshold used in the run
metrics_path = 'outputs/metrics.json'
spike_threshold = 2.0 # default
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        m = json.load(f)
        spike_threshold = m.get('spike_threshold', 2.0)

with open('outputs/backtest_summary.json', 'r') as f:
    bt = json.load(f)

print(f"--- Calibration Analysis (Target: 5-day Z-Score Max >= {spike_threshold}) ---")

xgb = bt['xgb_optimal']
rows = []
for regime, metrics in xgb.items():
    tiers = metrics.get('probability_tiers', {})
    for tier, v in tiers.items():
        rows.append({
            'Regime': regime,
            'Prob Tier': tier,
            'Win Rate': f"{v['win_rate']:.2%}",
            'Avg PnL': f"${v['avg_pnl']:,.2f}",
            'Avg Return': f"{v['avg_return']:.2%}",
            'N Trades': v['n_trades']
        })

df = pd.DataFrame(rows)
if not df.empty:
    df = df.sort_values(['Regime', 'Prob Tier'])
    print(df.to_string(index=False))
else:
    print("No trades found in backtest summary.")
