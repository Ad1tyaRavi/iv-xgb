# XGBoost-Based Options Volatility Spike Predictor

Predict imminent implied volatility (IV) spikes for SPY/SPX ATM options using engineered features from prices, realized vol, technicals, and (synthetic or real) Greeks.  
**Target:** whether IV rises ≥15% within the next 3 trading days.



## Project Structure

```
iv_spike_xgb_project/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py
│   ├── data_fetch.py
│   ├── features.py
│   ├── labeling.py
│   ├── models.py
│   ├── backtest.py
│   └── main.py
├── data/               # cached CSVs created at runtime
├── outputs/            # metrics, plots, feature importances
└── notebooks/
    └── IV_Spike_Predictor_guide.ipynb (optional; create later)
```

## Quickstart

1) **Install dependencies**
```
python -m venv .venv
source .venv/bin/activate  # windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) **Run the full pipeline (download → features → labels → models → reports)**
```
python -m src.main --ticker SPY --use-synthetic-greeks 1
```
- Use `--ticker SPX` if you have data access (yfinance provides SPX proxy as `^GSPC`; true index options chains require paid sources).  
- Set `--use-synthetic-greeks 0` only if you provide a historical options chain with IV/Greeks (place as `data/options_chain.csv` with the columns described below).

3) **Outputs**
- `outputs/metrics.json` — AUC, precision/recall/F1 for both Baseline(LogReg) and XGBoost
- `outputs/feature_importance_xgb.csv` — gain & weight importances
- `outputs/roc_curves.png` — ROC curves (Baseline vs XGBoost)
- `outputs/confusion_matrices.png` — pretty CM plots
- `outputs/backtest_summary.json` — #trades, win rate, average return, Sharpe
- `data/features_labeled.csv` — final training table (features + targets)

## Data

### Underlying
Downloaded via `yfinance` (daily OHLCV).

### Options chain
**Ideal:** Historical chain with columns:
```
date, expiry, strike, option_type, iv, delta, gamma, vega, theta, opt_price, under_price, volume, open_interest
```
Place file at `data/options_chain.csv`. If not available, **synthetic** Greeks/IV are generated realistically from underlyer features and realized vol.

## Features

**Underlying price features**
- returns, log_returns, high_low_range
- SMA10, SMA20, price_vs_sma10, price_vs_sma20
- RSI(14)
- volume_mean20, volume_ratio

**Vol/market features**
- realized_vol_(5, 10, 20, 30) = rolling std * sqrt(252)
- vix_proxy ≈ realized_vol_30 * 100
- vix_proxy_chg = pct_change

**Greeks (synthetic when needed)**
- iv ≈ realized_vol_20*(1.2 + 0.1*ε), boosted on large moves / bear regimes
- delta ≈ 0.5 + 0.2*tanh(price_vs_sma10*2)
- gamma ≈ 0.1*exp(-2*price_vs_sma10^2)
- vega ≈ iv*gamma*0.1
- theta ≈ -gamma*iv*close/365
- iv_vs_rv = iv / realized_vol_20 - 1
- iv_percentile_60 = rolling percentile rank of iv (60d)

**Regime feature**
- market_trend ∈ (-1, 0, 1) by price_vs_sma20 thresholds

## Target

- `iv_spike_3d`: 1 if `iv[t+3] / iv[t] - 1 ≥ 0.15`, else 0  
- also logs `iv_change_3d` for analysis

## Modeling

- **Baseline**: Logistic Regression (`StandardScaler`, `class_weight='balanced'`), test AUC
- **Main**: XGBoost classifier with grid over:
  - `max_depth: [3,4,5]`, `learning_rate: [0.05,0.1,0.2]`
  - `n_estimators: [100,200]`, `subsample: [0.8,0.9]`
  - `colsample_bytree: [0.8,0.9]`
- `TimeSeriesSplit(n_splits=3)` for CV, scored by ROC-AUC
- Leakage guards: **strict chronological split** (70%/30%).

## Trading Simulation

- **Signal**: predict spike today → enter at close, exit next day
- **Return proxy**: `iv[t+1] / iv[t] - 1` (stand‑in for ATM straddle sensitivity)
- Summary: #trades, hit rate, avg return, Sharpe(√252)

## Why this is hard (and valuable to learn)
- IV is anticipatory and reflexive; spikes are rare and regime‑dependent.
- Microstructure/holidays/earnings/Fed events cause nonstationarities.
- 

## Using Real Options Data (if you have it)
- Put `data/options_chain.csv` in the expected schema (above).
- Run with `--use-synthetic-greeks 0`. The rest of the pipeline is unchanged.

## Next Steps / Extensions
- Add event features (FOMC/earnings CPI/PPI) and test feature ablations.
- Try calibration (Platt / Isotonic) and threshold optimization by F1.
- Add SHAP analysis for model explainability.
- Extend backtest to actual straddle P&L if you have option quotes.

---

© 2025-09-16
