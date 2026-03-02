import json
import pandas as pd

with open('outputs/backtest_summary.json', 'r') as f:
    bt = json.load(f)

xgb = bt['xgb_optimal']
rows = []
for regime, metrics in xgb.items():
    tiers = metrics.get('probability_tiers', {})
    for tier, v in tiers.items():
        rows.append({
            'Regime': regime,
            'Tier': tier,
            'Win Rate': v['win_rate'],
            'Avg PnL': v['avg_pnl'],
            'Avg Return': v['avg_return'],
            'N Trades': v['n_trades']
        })

df = pd.DataFrame(rows)
df = df.sort_values(['Regime', 'Tier'])
print(df.to_string(index=False))
