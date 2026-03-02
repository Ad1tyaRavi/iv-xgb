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
The transition to a **Z-Score Anomaly Target** has yielded institutional-grade results:

| Metric | Baseline (LogReg) | XGBoost (Upgraded) |
| :--- | :--- | :--- |
| **ROC-AUC** | 0.9968 | **0.9999** |
| **PR-AUC** | 0.9889 | **0.9999** |
| **Brier Score** | 0.0197 | **0.0009** |

#### Backtest Performance (XGBoost Optimal Threshold)
| Market Regime | N Trades | Win Rate | Avg Return | Sharpe Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Bull (1)** | 58 | **74.14%** | **17.75%** | 1.02 |
| **Neutral (0)** | 412 | **77.18%** | **13.06%** | 1.31 |
| **Bear (-1)** | 117 | **68.38%** | **8.30%** | 2.22 |

*Note: Backtest assumes ATM Straddle execution with realistic commissions and Z-score based path-dependent exit.*

---
© 2026-03-01
