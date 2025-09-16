import pandas as pd, numpy as np

def make_labels(df: pd.DataFrame, spike_threshold: float = 0.15, lookahead_days: int = 3) -> pd.DataFrame:
    out = df.copy()
    out['iv_change_3d'] = out['iv'].shift(-lookahead_days) / out['iv'] - 1.0
    out['iv_spike_3d'] = (out['iv_change_3d'] >= spike_threshold).astype('float')
    # For robustness, we could add 1d and 5d too (not used by model unless desired)
    out['iv_change_1d'] = out['iv'].shift(-1) / out['iv'] - 1.0
    out['iv_change_5d'] = out['iv'].shift(-5) / out['iv'] - 1.0
    out['iv_spike_1d'] = (out['iv_change_1d'] >= spike_threshold).astype('float')
    out['iv_spike_5d'] = (out['iv_change_5d'] >= spike_threshold).astype('float')
    return out
