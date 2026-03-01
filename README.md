# XGBoost-Based Options Volatility Spike Predictor

Predict imminent implied volatility (IV) spikes for SPY/SPX ATM options using engineered features from prices, realized vol, technicals, and (synthetic or real) Greeks.  
**Target:** whether IV rises $\ge 15\%$ within the next 3 trading days.



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
    └── IV_Spike_Predictor_Analysis.ipynb # Rich analysis of the model performance
```

## Quickstart

1) **Install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate  # windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) **Run the full pipeline (download → features → labels → models → reports)**
```bash
python -m src.main
```
- By default, the project now expects high-quality local SPX options and underlying data located in `data/SPXdata/`. 
- If you don't have this data and wish to use `yfinance` to fetch underlying prices and synthesize options Greeks, you can run:
  ```bash
  python -m src.main --ticker SPY --use-synthetic-greeks 1
  ```

3) **Outputs**
- `outputs/metrics.json` — AUC, precision/recall/F1 for both Baseline(LogReg) and XGBoost
- `outputs/feature_importance_xgb.csv` — gain & weight importances
- `outputs/roc_curves.png` — ROC curves (Baseline vs XGBoost)
- `outputs/confusion_matrices_0.5.png` & `confusion_matrices_optimal.png` — pretty CM plots
- `outputs/backtest_summary.json` — #trades, win rate, average return, Sharpe
- `data/features_labeled.csv` — final training table (features + targets)

## Data

By default, the pipeline runs on a historical SPX dataset expected in `data/SPXdata/`. This includes:
- `SPXsecurites.csv`: Underlying index OHLCV.
- `SPXoptions.csv`: Daily options chains (used to extract ATM 30 DTE calls, bid/ask spreads, and Greeks).
- `SPXhistvol.csv`: Historical realized volatility.
- `zerocouponcurve.csv`: Daily risk-free rates.

*(Note: If this local data is unavailable, you can fall back to using `yfinance` for underlying prices and synthetically generating options Greeks by running with `--use-synthetic-greeks 1`.)*

## Features

**Underlying price features**
- returns, log_returns, high_low_range
- SMA10, SMA20, price_vs_sma10, price_vs_sma20
- RSI(14)
- volume_mean20, volume_ratio

**Vol/market features**
- `realized_vol_(5, 10, 20, 30)` = rolling std * $\sqrt{252}$
- Exact 30-day historical volatility (`hist_vol_30`)
- `vix_proxy` (based on `hist_vol_30` or Rogers-Satchell calculation)
- vix_proxy_chg = pct_change

**Greeks & Pricing (from real SPX data)**
- `iv`, `delta`, `gamma`, `vega`, `theta` for 30 DTE ATM calls
- iv_vs_rv = iv / hist_vol_30 - 1
- iv_percentile_60 = rolling percentile rank of iv (60d)
*(Synthetic generation is available as a fallback if real data isn't provided).*

**Regime feature**
- `market_trend` $\in$ (-1, 0, 1) by price_vs_sma20 thresholds

## Target

- `iv_spike_3d`: 1 if $iv_{t+3} / iv_t - 1 \ge 0.15$, else 0  
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

We run a high-fidelity backtest to see how the model would perform in the real market:
- **Signal**: Predict an IV spike today.
- **Execution**: Buy an ATM 30 DTE Call option at the historical **Best Offer** (Ask). 
- **Holding Period**: Hold for exactly 3 days to match our prediction window.
- **Exit**: Sell at the historical **Best Bid** (crossing the spread).
- **Summary**: We break down the number of trades, win rate, average return, and Sharpe proxy across different market regimes (Bull/Bear/Neutral).

## Why this is hard (and valuable to learn)
- IV is anticipatory and reflexive; spikes are rare and regime‑dependent.
- Microstructure/holidays/earnings/Fed events cause nonstationarities.

## Next Steps / Extensions
- Add macro event features (FOMC, CPI, PPI) to catch structural volatility events.
- Explore calibration (Platt / Isotonic) to get better probability estimates.
- Add SHAP analysis to make the XGBoost model's decisions more interpretable.

---

© 2025-09-16
