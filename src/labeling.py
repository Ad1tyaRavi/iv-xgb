import pandas as pd, numpy as np

def make_labels(df: pd.DataFrame, spike_threshold: float = 0.15, lookahead_days: int = 3) -> pd.DataFrame:
    out = df.copy()
    
    # 3d spike
    future_ivs_3d = [out['iv'].shift(-i) for i in range(1, lookahead_days + 1)]
    max_future_iv_3d = pd.concat(future_ivs_3d, axis=1).max(axis=1)
    out['iv_change_3d'] = max_future_iv_3d / out['iv'] - 1
    out['iv_spike_3d'] = (out['iv_change_3d'] >= spike_threshold).astype('float')

    # 1d spike
    out['iv_change_1d'] = out['iv'].shift(-1) / out['iv'] - 1.0
    out['iv_spike_1d'] = (out['iv_change_1d'] >= spike_threshold).astype('float')

    # 5d spike
    future_ivs_5d = [out['iv'].shift(-i) for i in range(1, 5 + 1)]
    max_future_iv_5d = pd.concat(future_ivs_5d, axis=1).max(axis=1)
    out['iv_change_5d'] = max_future_iv_5d / out['iv'] - 1
    out['iv_spike_5d'] = (out['iv_change_5d'] >= spike_threshold).astype('float')
    
    return out