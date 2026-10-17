from pathlib import Path

import pandas as pd


def exercise():
    """Load the vendored seaborn-data/exercise.csv snapshot without network I/O."""
    # Source: https://github.com/mwaskom/seaborn-data/blob/master/exercise.csv
    return pd.read_csv(Path(__file__).parent / "data" / "exercise.csv")
