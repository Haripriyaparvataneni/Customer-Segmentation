"""RFM feature engineering.

Computes Recency (days since last purchase), Frequency (order count),
and Monetary (total spend) per customer from the transactions table.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Return an RFM DataFrame with one row per customer.

    Expected columns: customer_id, date, amount.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    snapshot = df["date"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("customer_id")
        .agg(
            Recency=("date", lambda s: (snapshot - s.max()).days),
            Frequency=("date", "count"),
            Monetary=("amount", "sum"),
        )
        .reset_index()
    )
    rfm["Monetary"] = rfm["Monetary"].round(2)
    return rfm


if __name__ == "__main__":
    transactions = pd.read_csv(ROOT / "data/transactions.csv")
    rfm = compute_rfm(transactions)
    out = ROOT / "data/rfm.csv"
    rfm.to_csv(out, index=False)
    print(f"Wrote {out}: {len(rfm):,} customers")
    print(rfm.describe().round(1).to_string())
