"""Generate synthetic retail transactions (portfolio demo, not real data).

Creates data/transactions.csv: ~1 year of transactions for 1,000 customers.
Customers follow four archetypes so the clustering step finds meaningful
structure: loyal regulars, big spenders, new customers, and lapsed customers.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SEED = 42
N_CUSTOMERS = 1000
N_DAYS = 365

# (share, avg_orders_per_year, avg_order_value, recency_bias)
# recency_bias shifts purchase dates toward the end (active) or start (lapsed).
ARCHETYPES = {
    "loyal": (0.30, 18, 60, 0.0),
    "big_spender": (0.20, 6, 220, 0.0),
    "new": (0.25, 3, 55, 1.0),      # purchases concentrated near the end
    "lapsed": (0.25, 8, 70, -1.0),  # purchases concentrated near the start
}


def generate(n_customers: int = N_CUSTOMERS, n_days: int = N_DAYS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=n_days - 1)
    rows = []
    txn_id = 0
    for c in range(n_customers):
        archetype = rng.choice(
            list(ARCHETYPES), p=[ARCHETYPES[a][0] for a in ARCHETYPES]
        )
        _, avg_orders, avg_value, bias = ARCHETYPES[archetype]
        n_orders = max(1, int(rng.poisson(avg_orders)))
        # Recency bias: warp uniform [0,1) toward recent (1.0) or old (-1.0).
        u = rng.random(n_orders)
        if bias > 0:
            u = u**0.4
        elif bias < 0:
            u = 1 - (1 - u) ** 0.4
        dates = start + pd.to_timedelta((u * n_days).astype(int), unit="D")
        amounts = np.maximum(rng.normal(avg_value, avg_value * 0.35, n_orders), 5)
        for d, a in zip(dates, amounts):
            txn_id += 1
            rows.append((txn_id, f"C{c:04d}", d.date().isoformat(), round(float(a), 2)))
    return pd.DataFrame(rows, columns=["transaction_id", "customer_id", "date", "amount"])


if __name__ == "__main__":
    df = generate()
    out = ROOT / "data/transactions.csv"
    df.to_csv(out, index=False)
    print(
        f"Wrote {out}: {len(df):,} transactions, "
        f"{df['customer_id'].nunique()} customers"
    )
