# Customer Segmentation with RFM + K-Means

I wanted a simple, end-to-end segmentation project, so this one takes raw transactions, builds RFM features, and groups customers with K-Means.

The transactions are synthetic — I wrote `data/generate_data.py` to make them, roughly a year of orders for 1,000 customers. No real customer data anywhere in this repo.

## How it works

The generator mixes four types of customers on purpose (loyal regulars, big spenders, new customers, and people who stopped buying), so there is actually something for the clustering to find.

From the transactions I calculate the usual RFM values for each customer:

- **Recency** — days since their last order
- **Frequency** — how many orders they placed
- **Monetary** — total amount spent

Then K-Means runs on the scaled RFM features. I pick k using the elbow method (it landed on 4), profile each cluster, and give the clusters names a marketing person would actually use instead of "cluster 0, cluster 1...".

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python data/generate_data.py   # make the transactions
python src/rfm.py              # build the RFM table
python src/cluster.py          # cluster + label the segments
python src/visualize.py        # plots into figures/
```

## What came out

| Segment | Size | Avg Recency (days) | Avg Frequency | Avg Monetary ($) |
|---|---|---|---|---|
| Champions | 294 | 23.3 | 18.4 | 1,113.70 |
| New Customers | 322 | 41.3 | 4.2 | 315.50 |
| At Risk | 146 | 45.5 | 7.2 | 1,586.00 |
| Hibernating | 238 | 169.6 | 6.3 | 496.60 |

- **Champions** — bought recently, buy often, spend the most. Just keep them happy.
- **New Customers** — recent, but only a couple of orders so far.
- **At Risk** — spent a lot in the past but haven't bought in a while. These are the win-back campaign people.
- **Hibernating** — quiet for months. Low priority.

The "At Risk" group having the highest average spend surprised me at first, but it makes sense — big spenders who drift away drag that average up.

`visualize.py` saves two plots to `figures/`: RFM distribution histograms, and a Recency vs Monetary scatter with the segments in different colours.

## Project structure

```
customer-segmentation/
├── data/
│   └── generate_data.py     # makes the synthetic transactions
├── src/
│   ├── rfm.py               # Recency / Frequency / Monetary features
│   ├── cluster.py           # K-Means, elbow check, segment labels
│   └── visualize.py         # the two plots
├── figures/                 # plots land here (created by visualize.py)
├── requirements.txt
└── .gitignore
```

The generated CSV files aren't in git — run the generator and you'll have them locally.
