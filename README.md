# XGBoost-Based Options Volatility Spike Predictor (Z-Score Anomaly Edition)

Predict statistical anomalies in implied volatility (IV) for SPX ATM options using engineered features from prices, realized vol, technicals, and Greeks.

**Target:** Statistical anomaly defined as a **5-day rolling Z-Score maximum $\ge 2.0$**. This forces the model to predict genuine regime shifts normalized for different market environments.

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
├── outputs/            # metrics, SHAP plots, backtest results
└── notebooks/
    └── IV_Spike_Predictor_Analysis.ipynb # Comprehensive analysis & SHAP visualizations
```

## Quickstart

1) **Install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate  # windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) **Run the full pipeline (download → features → labels → models → SHAP → reports)**
```bash
python -m src.main
```

3) **Outputs**
- `outputs/metrics.json` — AUC, PR-AUC, and **Brier Score** (Calibration)
- `outputs/shap_summary_xgb.png` — **SHAP Summary Plot** showing exact non-linear decision boundaries
- `outputs/feature_importance_xgb.csv` — SHAP-based feature importance
- `outputs/roc_curves.png` & `outputs/pr_curves.png` — Performance visualizations
- `outputs/backtest_summary.json` — Path-dependent simulation results (PnL, Sharpe, Max DD)

## Core Upgrades

### 1. Statistical Anomaly Targeting
Instead of a naive percentage move, we use a **rolling Z-Score** calculation:
- $Z = (IV_t - \mu_{20d}) / \sigma_{20d}$
- **Label = 1** if $\max(Z_{t+1}, \dots, Z_{t+5}) \ge 2.0$
- This allows the model to learn what constitutes a "spike" relative to the current volatility regime (e.g., a 2% move in a 10-vol environment vs a 10% move in a 40-vol environment).

### 2. Probability Calibration (Brier Score)
We evaluate model confidence using the **Brier Score** ($BS = \frac{1}{N} \sum (f_i - o_i)^2$). This ensures that a 70% probability output from XGBoost actually corresponds to a 70% historical frequency of spikes, which is critical for position sizing.

### 3. SHAP Interpretability
We have replaced native feature weights with **SHapley Additive exPlanations (SHAP)**. This reveals:
- Which features (e.g., Volatility Skew, RSI, or Gamma) are the primary drivers of an IV expansion.
- The non-linear interaction between features (e.g., how low RSI combined with high Skew triggers a regime shift).

## Trading Simulation (Backtest)

Institutional-grade, path-dependent engine:
- **Strategy**: ATM Straddle execution with $0.65/leg commissions.
- **Risk Management**: 3 concurrent trade slots (portfolio diversification).
- **Oracle Exit**: Immediate exit if the Z-Score threshold (2.0) is hit within the 5-day window.
- **Stratification**: Results are broken down by **probability tiers** and market regimes (Bull/Bear/Neutral).

### Performance Summary (Latest Run)
The model provides realistic, non-cheating predictive power for IV regime shifts:

| Metric | Baseline (LogReg) | XGBoost (Validated) |
| :--- | :--- | :--- |
| **ROC-AUC** | 0.7029 | **0.7176** |
| **PR-AUC** | 0.3845 | **0.3736** |
| **Brier Score** | 0.2466 | **0.2008** |

#### Backtest Performance (XGBoost Optimal Threshold)
| Market Regime | N Trades | Win Rate | Total Return | Sharpe Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Bull (1)** | 29 | **48.28%** | **26.16%** | 0.63 |
| **Neutral (0)** | 587 | **41.57%** | **-389.12%** | -0.16 |
| **Bear (-1)** | 148 | **56.08%** | **35.87%** | 0.56 |

*Note: Results reflect path-dependent simulation with $0.65/leg commissions. Neutral regimes show high decay costs due to higher trade frequency without offsetting spikes.*

*Note: Backtest assumes ATM Straddle execution with realistic commissions and Z-score based path-dependent exit.*

---
© 2026-03-01
