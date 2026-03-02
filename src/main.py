import os, json, argparse, numpy as np, pandas as pd, datetime as dt
from .config import Config
from .data_fetch import fetch_underlying_ohlcv, fetch_local_spx_data
from .features import add_underlying_features, synthesize_greeks, finalize_feature_table
from .labeling import make_labels
from .models import chronological_split, train_baseline_logreg, train_xgb, plot_roc_curves, plot_confusions, feature_importance_xgb, plot_pr_curves, find_optimal_threshold
from .backtest import realistic_iv_signal_backtest
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit

def run(cfg: Config):
    os.makedirs('data', exist_ok=True)
    os.makedirs(cfg.outputs_dir, exist_ok=True)

    # 1) Data
    data = fetch_local_spx_data()
    data = data.sort_values('date')

    # 2) Features
    features = add_underlying_features(data)
    if cfg.use_synthetic_greeks:
        rng = np.random.default_rng(42)
        features = synthesize_greeks(features, rng)
    else:
        rv_base = features['hist_vol_30'] if 'hist_vol_30' in features.columns else features['rogers_satchell_vol_20']
        features['iv_vs_rv'] = features['iv'] / rv_base - 1
        features['iv_percentile_60'] = features['iv'].rolling(60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x)==60 else np.nan, raw=False
        )

    # 3) Labels
    labeled = make_labels(features, spike_threshold=cfg.spike_threshold, lookahead_days=cfg.lookahead_days)

    # 4) Final table
    final = finalize_feature_table(labeled)

    # We drop trade_ret and synthetic pricing since we use path-dependent daily simulation now.
    
    final = final.dropna(subset=['iv_spike_3d', 'market_trend', 'best_bid', 'best_offer', 'iv'])
    
    X_cols = final.drop(columns=['date','close','iv_spike_3d','iv_change_3d','iv_spike_1d','iv_spike_5d','iv_change_1d','iv_change_5d', 'best_bid', 'best_offer'], errors='ignore').columns
    
    final = final.dropna(subset=X_cols)
    final.to_csv(cfg.features_csv, index=False)

    # --- Walk-Forward Evaluation ---
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=cfg.lookahead_days)

    all_y_true = []
    all_p_base = []
    all_p_xgb = []
    all_test_dfs = []

    for i, (train_idx, test_idx) in enumerate(tscv.split(final)):
        print(f"--- Fold {i+1}/{n_splits} ---")
        train_df = final.iloc[train_idx]
        test_df = final.iloc[test_idx]

        y_train = train_df['iv_spike_3d'].values
        y_test = test_df['iv_spike_3d'].values
        X_train = train_df[X_cols].values
        X_test = test_df[X_cols].values

        base = train_baseline_logreg(X_train, y_train, X_test, y_test)
        xgb = train_xgb(X_train, y_train, X_test, y_test, lookahead_days=cfg.lookahead_days)

        # Optimize threshold on training fold to eliminate lookahead bias
        train_p_xgb = xgb['model'].predict_proba(X_train)[:, 1]
        opt_thr_xgb = find_optimal_threshold(y_train, train_p_xgb)

        all_y_true.extend(y_test)
        all_p_base.extend(base['proba'])
        all_p_xgb.extend(xgb['proba'])
        
        test_df = test_df.copy()
        test_df['proba_baseline'] = base['proba']
        test_df['proba_xgb'] = xgb['proba']
        test_df['optimal_threshold'] = opt_thr_xgb
        all_test_dfs.append(test_df)

    # --- Aggregate Results ---
    all_y_true = np.array(all_y_true)
    all_p_base = np.array(all_p_base)
    all_p_xgb = np.array(all_p_xgb)
    all_test_df = pd.concat(all_test_dfs)
    
    # Calculate thresholded predictions using per-fold optimal threshold
    yhat_opt_xgb = (all_test_df['proba_xgb'] >= all_test_df['optimal_threshold']).astype(int)

    metrics = {
        'baseline_auc': float(roc_auc_score(all_y_true, all_p_base)),
        'xgb_auc': float(roc_auc_score(all_y_true, all_p_xgb)),
        'baseline_pr_auc': float(average_precision_score(all_y_true, all_p_base)),
        'xgb_pr_auc': float(average_precision_score(all_y_true, all_p_xgb)),
        'xgb_avg_optimal_threshold': float(all_test_df['optimal_threshold'].mean()),
        'classification_report_threshold_0.5': {
            'baseline': None,
            'xgb': None
        },
        'classification_report_threshold_optimal': {
            'xgb': None
        }
    }

    def prf(y_true, yhat):
        p,r,f,_ = precision_recall_fscore_support(y_true, yhat, average='binary', zero_division=0)
        return {'precision': float(p), 'recall': float(r), 'f1': float(f)}

    metrics['classification_report_threshold_0.5']['baseline'] = prf(all_y_true, (all_p_base>=0.5).astype(int))
    metrics['classification_report_threshold_0.5']['xgb'] = prf(all_y_true, (all_p_xgb>=0.5).astype(int))
    metrics['classification_report_threshold_optimal']['xgb'] = prf(all_y_true, yhat_opt_xgb)

    with open(os.path.join(cfg.outputs_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_roc_curves(all_y_true, all_p_base, all_p_xgb, ['Baseline','XGBoost'], os.path.join(cfg.outputs_dir,'roc_curves.png'))
    plot_pr_curves(all_y_true, all_p_base, all_p_xgb, ['Baseline','XGBoost'], os.path.join(cfg.outputs_dir,'pr_curves.png'))
    plot_confusions(all_y_true, all_p_base, all_p_xgb, os.path.join(cfg.outputs_dir,'confusion_matrices_0.5.png'))
    
    avg_opt_thr = all_test_df['optimal_threshold'].mean()
    plot_confusions(all_y_true, all_p_base, all_p_xgb, os.path.join(cfg.outputs_dir,'confusion_matrices_optimal.png'), thr=avg_opt_thr)

    imp = feature_importance_xgb(xgb['model'], list(X_cols))
    imp.to_csv(os.path.join(cfg.outputs_dir, 'feature_importance_xgb.csv'), index=False)

    # Backtest
    bt_base = realistic_iv_signal_backtest(all_test_df, 'proba_baseline', use_dynamic_threshold=False, fixed_threshold=0.5, group_by_col='market_trend')
    bt_xgb_05 = realistic_iv_signal_backtest(all_test_df, 'proba_xgb', use_dynamic_threshold=False, fixed_threshold=0.5, group_by_col='market_trend')
    bt_xgb_opt = realistic_iv_signal_backtest(all_test_df, 'proba_xgb', use_dynamic_threshold=True, group_by_col='market_trend')

    with open(os.path.join(cfg.outputs_dir, 'backtest_summary.json'), 'w') as f:
        json.dump({'baseline': bt_base, 'xgb_0.5': bt_xgb_05, 'xgb_optimal': bt_xgb_opt}, f, indent=2)

    print('Done. Key results:')
    print(json.dumps({'AUC': {'baseline': metrics['baseline_auc'], 'xgb': metrics['xgb_auc']},
                      'PR-AUC': {'baseline': metrics['baseline_pr_auc'], 'xgb': metrics['xgb_pr_auc']}}, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', type=str, default='SPX')
    p.add_argument('--start', type=str, default='2006-01-01')
    p.add_argument('--end', type=str, default=None)
    p.add_argument('--use-synthetic-greeks', type=int, default=0)
    args = p.parse_args()

    cfg = Config(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        use_synthetic_greeks=bool(args.use_synthetic_greeks)
    )
    run(cfg)
