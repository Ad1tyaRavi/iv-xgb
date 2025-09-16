import os, json, argparse, numpy as np, pandas as pd, datetime as dt
from .config import Config
from .data_fetch import fetch_underlying_ohlcv
from .features import add_underlying_features, synthesize_greeks, finalize_feature_table
from .labeling import make_labels
from .models import chronological_split, train_baseline_logreg, train_xgb, plot_roc_curves, plot_confusions, feature_importance_xgb
from .backtest import simple_iv_signal_backtest

def run(cfg: Config):
    os.makedirs('data', exist_ok=True)
    os.makedirs(cfg.outputs_dir, exist_ok=True)

    # 1) Data
    under = fetch_underlying_ohlcv(cfg.ticker, cfg.start, cfg.end)

    # 2) Features
    features = add_underlying_features(under)
    rng = np.random.default_rng(42)
    features = synthesize_greeks(features, rng) if cfg.use_synthetic_greeks else features

    # 3) Labels
    labeled = make_labels(features, spike_threshold=cfg.spike_threshold, lookahead_days=cfg.lookahead_days)

    # 4) Final table
    final = finalize_feature_table(labeled) # This is a pass-through now
    
    # Drop rows where the label is NaN, as we cannot train on them.
    # Features with NaNs will be handled by the imputation pipeline in the model.
    final = final.dropna(subset=['iv_spike_3d'])
    final.to_csv(cfg.features_csv, index=False)

    # Train/test split (chronological)
    y = final['iv_spike_3d']
    X = final.drop(columns=['date','close','iv_spike_3d','iv_change_3d','iv_spike_1d','iv_spike_5d','iv_change_1d','iv_change_5d'], errors='ignore')
    train_df, test_df = chronological_split(final, cfg.train_frac)
    y_train = train_df['iv_spike_3d'].values
    y_test = test_df['iv_spike_3d'].values
    X_train = train_df[X.columns].values
    X_test = test_df[X.columns].values

    # 5) Baseline
    base = train_baseline_logreg(X_train, y_train, X_test, y_test)

    # 6) XGBoost
    xgb = train_xgb(X_train, y_train, X_test, y_test)

    # 7) Evaluation artifacts
    metrics = {
        'baseline_auc': float(base['auc']),
        'xgb_auc': float(xgb['auc']),
        'classification_report_threshold_0.5': {
            'baseline': None,
            'xgb': None
        }
    }

    # Thresholded metrics
    from sklearn.metrics import precision_recall_fscore_support
    def prf(y_true, proba, thr=0.5):
        yhat = (proba>=thr).astype(int)
        p,r,f,_ = precision_recall_fscore_support(y_true, yhat, average='binary', zero_division=0)
        return {'precision': float(p), 'recall': float(r), 'f1': float(f)}

    metrics['classification_report_threshold_0.5']['baseline'] = prf(y_test, base['proba'])
    metrics['classification_report_threshold_0.5']['xgb'] = prf(y_test, xgb['proba'])

    with open(os.path.join(cfg.outputs_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_roc_curves(y_test, base['proba'], xgb['proba'], ['Baseline','XGBoost'], os.path.join(cfg.outputs_dir,'roc_curves.png'))
    plot_confusions(y_test, base['proba'], xgb['proba'], os.path.join(cfg.outputs_dir,'confusion_matrices.png'))

    # Feature importances
    imp = feature_importance_xgb(xgb['model'], list(X.columns))
    imp.to_csv(os.path.join(cfg.outputs_dir, 'feature_importance_xgb.csv'), index=False)

    # Backtest (simple)
    test_df = test_df.copy()
    test_df['proba_baseline'] = base['proba']
    test_df['proba_xgb'] = xgb['proba']
    bt_base = simple_iv_signal_backtest(test_df, 'proba_baseline', 0.5)
    bt_xgb = simple_iv_signal_backtest(test_df, 'proba_xgb', 0.5)

    with open(os.path.join(cfg.outputs_dir, 'backtest_summary.json'), 'w') as f:
        json.dump({'baseline': bt_base, 'xgb': bt_xgb}, f, indent=2)

    print('Done. Key results:')
    print(json.dumps({'AUC': {'baseline': base['auc'], 'xgb': xgb['auc']},
                      'Backtest': {'baseline': bt_base, 'xgb': bt_xgb}}, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', type=str, default='SPY')
    p.add_argument('--start', type=str, default='2010-01-01')
    p.add_argument('--end', type=str, default=None)
    p.add_argument('--use-synthetic-greeks', type=int, default=1)
    args = p.parse_args()

    cfg = Config(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        use_synthetic_greeks=bool(args.use_synthetic_greeks)
    )
    run(cfg)
