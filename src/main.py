import os, json, argparse, numpy as np, pandas as pd, datetime as dt
from .config import Config
from .data_fetch import fetch_underlying_ohlcv
from .features import add_underlying_features, synthesize_greeks, finalize_feature_table
from .labeling import make_labels
from .models import chronological_split, train_baseline_logreg, train_xgb, plot_roc_curves, plot_confusions, feature_importance_xgb, plot_pr_curves, find_optimal_threshold
from .backtest import realistic_iv_signal_backtest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

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
    final = finalize_feature_table(labeled)
    final = final.dropna(subset=['iv_spike_3d', 'market_trend'])
    final.to_csv(cfg.features_csv, index=False)

    # --- Walk-Forward Evaluation ---
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=cfg.lookahead_days)

    all_y_true = []
    all_p_base = []
    all_p_xgb = []
    all_test_dfs = []

    X_cols = final.drop(columns=['date','close','iv_spike_3d','iv_change_3d','iv_spike_1d','iv_spike_5d','iv_change_1d','iv_change_5d'], errors='ignore').columns

    for i, (train_idx, test_idx) in enumerate(tscv.split(final)):
        print(f"--- Fold {i+1}/{n_splits} ---")
        train_df = final.iloc[train_idx]
        test_df = final.iloc[test_idx]

        y_train = train_df['iv_spike_3d'].values
        y_test = test_df['iv_spike_3d'].values
        X_train = train_df[X_cols].values
        X_test = test_df[X_cols].values

        # Train models
        base = train_baseline_logreg(X_train, y_train, X_test, y_test)
        xgb = train_xgb(X_train, y_train, X_test, y_test, lookahead_days=cfg.lookahead_days)

        # Collect predictions
        all_y_true.extend(y_test)
        all_p_base.extend(base['proba'])
        all_p_xgb.extend(xgb['proba'])
        
        # Collect test data for backtesting
        test_df = test_df.copy()
        test_df['proba_baseline'] = base['proba']
        test_df['proba_xgb'] = xgb['proba']
        all_test_dfs.append(test_df)

    # --- Aggregate Results ---
    all_y_true = np.array(all_y_true)
    all_p_base = np.array(all_p_base)
    all_p_xgb = np.array(all_p_xgb)
    all_test_df = pd.concat(all_test_dfs)

    # Find optimal threshold for XGBoost on all out-of-sample predictions
    optimal_thr_xgb = find_optimal_threshold(all_y_true, all_p_xgb)

    metrics = {
        'baseline_auc': float(roc_auc_score(all_y_true, all_p_base)),
        'xgb_auc': float(roc_auc_score(all_y_true, all_p_xgb)),
        'baseline_pr_auc': float(average_precision_score(all_y_true, all_p_base)),
        'xgb_pr_auc': float(average_precision_score(all_y_true, all_p_xgb)),
        'xgb_optimal_threshold': float(optimal_thr_xgb),
        'classification_report_threshold_0.5': {
            'baseline': None,
            'xgb': None
        },
        'classification_report_threshold_optimal': {
            'xgb': None
        }
    }

    # Thresholded metrics
    from sklearn.metrics import precision_recall_fscore_support
    def prf(y_true, proba, thr=0.5):
        yhat = (proba>=thr).astype(int)
        p,r,f,_ = precision_recall_fscore_support(y_true, yhat, average='binary', zero_division=0)
        return {'precision': float(p), 'recall': float(r), 'f1': float(f)}

    metrics['classification_report_threshold_0.5']['baseline'] = prf(all_y_true, all_p_base)
    metrics['classification_report_threshold_0.5']['xgb'] = prf(all_y_true, all_p_xgb)
    metrics['classification_report_threshold_optimal']['xgb'] = prf(all_y_true, all_p_xgb, thr=optimal_thr_xgb)

    with open(os.path.join(cfg.outputs_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_roc_curves(all_y_true, all_p_base, all_p_xgb, ['Baseline','XGBoost'], os.path.join(cfg.outputs_dir,'roc_curves.png'))
    plot_pr_curves(all_y_true, all_p_base, all_p_xgb, ['Baseline','XGBoost'], os.path.join(cfg.outputs_dir,'pr_curves.png'))
    plot_confusions(all_y_true, all_p_base, all_p_xgb, os.path.join(cfg.outputs_dir,'confusion_matrices_0.5.png'))
    plot_confusions(all_y_true, all_p_base, all_p_xgb, os.path.join(cfg.outputs_dir,'confusion_matrices_optimal.png'), thr=optimal_thr_xgb)

    # Feature importances (from the last fold's XGB model)
    imp = feature_importance_xgb(xgb['model'], list(X_cols))
    imp.to_csv(os.path.join(cfg.outputs_dir, 'feature_importance_xgb.csv'), index=False)

    # Backtest
    bt_base = realistic_iv_signal_backtest(all_test_df, 'proba_baseline', 0.5, group_by_col='market_trend')
    bt_xgb_05 = realistic_iv_signal_backtest(all_test_df, 'proba_xgb', 0.5, group_by_col='market_trend')
    bt_xgb_opt = realistic_iv_signal_backtest(all_test_df, 'proba_xgb', optimal_thr_xgb, group_by_col='market_trend')

    with open(os.path.join(cfg.outputs_dir, 'backtest_summary.json'), 'w') as f:
        json.dump({'baseline': bt_base, 'xgb_0.5': bt_xgb_05, 'xgb_optimal': bt_xgb_opt}, f, indent=2)

    print('Done. Key results:')
    print(json.dumps({'AUC': {'baseline': metrics['baseline_auc'], 'xgb': metrics['xgb_auc']},
                      'PR-AUC': {'baseline': metrics['baseline_pr_auc'], 'xgb': metrics['xgb_pr_auc']},
                      'Backtest': {'baseline': bt_base, 'xgb_0.5': bt_xgb_05, 'xgb_optimal': bt_xgb_opt}}, indent=2))

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