"""K-Means customer segmentation on RFM features.

Selects k with a simple elbow heuristic (max distance from the chord
between the first and last inertia points), clusters the scaled RFM
features, profiles each cluster, and assigns business-friendly labels
like "Champions" or "At Risk".
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent


def elbow_k(inertias: dict[int, float]) -> int:
    """Pick the elbow: k with max distance from the chord (k_min, k_max)."""
    ks = np.array(sorted(inertias))
    ys = np.array([inertias[k] for k in ks])
    # Normalize to unit square so distance is scale-free.
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (ys - ys.min()) / (ys.max() - ys.min())
    # Distance from point to chord between first and last.
    chord = np.array([x[-1] - x[0], y[-1] - y[0]])
    chord /= np.linalg.norm(chord)
    dists = np.abs((y - y[0]) * chord[0] - (x - x[0]) * chord[1])
    return int(ks[np.argmax(dists)])


def label_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Assign a business label to each cluster from its RFM profile.

    profile: DataFrame indexed by cluster with Recency/Frequency/Monetary means.
    """
    r, f, m = profile["Recency"], profile["Frequency"], profile["Monetary"]
    labels = {}
    for c in profile.index:
        recent, freq, money = r[c] < r.median(), f[c] > f.median(), m[c] > m.median()
        if recent and freq and money:
            labels[c] = "Champions"
        elif recent and freq:
            labels[c] = "Potential Loyalists"
        elif recent and not freq and not money:
            labels[c] = "New Customers"
        elif not recent and freq and money:
            labels[c] = "At Risk"
        elif not recent and not freq and not money:
            labels[c] = "Hibernating"
        elif not recent and money:
            labels[c] = "Need Attention"
        else:
            labels[c] = "Regulars"
    return labels


def main() -> None:
    rfm = pd.read_csv(ROOT / "data/rfm.csv")
    X = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    inertias = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias[k] = km.inertia_
    k = elbow_k(inertias)
    print(f"Inertias: { {kk: round(v) for kk, v in inertias.items()} }")
    print(f"Selected k={k} (elbow)")

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    rfm["cluster"] = km.fit_predict(X)

    profile = rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]].mean().round(1)
    profile["size"] = rfm["cluster"].value_counts().sort_index()
    labels = label_clusters(profile)
    rfm["segment"] = rfm["cluster"].map(labels)
    profile["segment"] = profile.index.map(labels)
    print("\nCluster profiles:")
    print(profile[["segment", "size", "Recency", "Frequency", "Monetary"]].to_string())

    rfm.to_csv(ROOT / "data/labeled_rfm.csv", index=False)
    print(f"\nWrote {ROOT / 'data/labeled_rfm.csv'}")


if __name__ == "__main__":
    main()
