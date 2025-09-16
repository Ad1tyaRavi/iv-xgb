from dataclasses import dataclass

@dataclass
class Config:
    ticker: str = "SPY"
    start: str = "2010-01-01"
    end: str = None  # defaults to today
    use_synthetic_greeks: bool = True
    features_csv: str = "data/features_labeled.csv"
    options_csv: str = "data/options_chain.csv"  # only needed if real data provided
    outputs_dir: str = "outputs"
    spike_threshold: float = 0.05
    lookahead_days: int = 3
    train_frac: float = 0.7
