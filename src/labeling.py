import pandas as pd, numpy as np

def make_labels(df: pd.DataFrame, spike_threshold: float = 2.0, lookahead_days: int = 5) -> pd.DataFrame:
    """
    Redefines target as a rolling Z-Score maximum over a 5-day window.
    Captures statistical anomalies normalized for different market regimes.
    """
    out = df.copy()

    # Calculate rolling mean and std of IV over a trailing window (e.g., 20 days)
    # to define the "normal" regime before the prediction point
    rolling_mean = out['iv'].rolling(window=20).mean()
    rolling_std = out['iv'].rolling(window=20).std()

    # Calculate Z-Score of IV
    # Adding a small epsilon to std to avoid division by zero
    out['iv_zscore'] = (out['iv'] - rolling_mean) / (rolling_std + 1e-6)

    # Target: Maximum Z-Score achieved in the next 'lookahead_days'
    future_z = [out['iv_zscore'].shift(-i) for i in range(1, lookahead_days + 1)]
    max_future_z = pd.concat(future_z, axis=1).max(axis=1)

    out['max_future_z'] = max_future_z
    # Target 1 if max future Z-score >= threshold, else 0
    out['iv_spike_3d'] = (out['max_future_z'] >= spike_threshold).astype('float') # Name kept for pipeline compatibility

    # Add some auxiliary info
    out['iv_change_3d'] = out['iv'].shift(-lookahead_days) / out['iv'] - 1 # Adjusted for lookahead_days

    return out