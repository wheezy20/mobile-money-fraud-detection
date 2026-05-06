"""Step 1: Load raw PaySim transaction data."""
import pandas as pd
from typing_extensions import Annotated
from zenml import step


@step
def load_data(data_path: str = "data/raw/transactions.csv") -> Annotated[pd.DataFrame, "raw_data"]:
    """Load mobile money transaction data from CSV."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} transactions, {df['isFraud'].sum():,} fraudulent")
    print(f"Fraud rate: {df['isFraud'].mean():.4%}")
    return df
