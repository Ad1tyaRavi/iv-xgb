from dataclasses import dataclass

@dataclass
class Config:
    ticker: str = "SPX"  # Switched to SPX for local data
    start: str = "2006-01-01"  # Data starts early 2006
    end: str = None
    use_synthetic_greeks: bool = False  # Set to False to use local data greeks
    features_csv: str = "data/features_labeled.csv"
    options_csv: str = "data/SPXdata/SPXoptions.csv"
    securities_csv: str = "data/SPXdata/SPXsecurites.csv"
    outputs_dir: str = "outputs"
    spike_threshold: float = 2.0
    lookahead_days: int = 5
    train_frac: float = 0.7
