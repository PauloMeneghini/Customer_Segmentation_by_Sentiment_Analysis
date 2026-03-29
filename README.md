# Customer Segmentation via RFM Analysis & K-Means Clustering

> Unsupervised machine learning pipeline to segment 95,000+ e-commerce customers into behavioral profiles using the RFM framework — enabling data-driven retention and marketing strategies.

---

## Business Problem

Most e-commerce companies treat all customers the same. This project challenges that assumption.

Using transactional data from **Olist** (the largest Brazilian marketplace dataset on Kaggle), this pipeline automatically classifies every customer into one of three strategic segments:

| Segment | Profile | Recommended Action |
|---|---|---|
| **VIP** | High spenders, recent buyers | Loyalty rewards, early access |
| **Regular** | Average engagement | Upsell campaigns, cross-sell |
| **At Risk** | Haven't bought in 300+ days | Win-back email campaigns |

---

## Architecture

The project follows a **modular pipeline design** — each responsibility is isolated in its own module, making it easy to test, maintain and extend.

```
main.py                  ← Orchestrator. Run this.
│
├── data_prep.py         ← Data ingestion, filtering & type casting
├── feature_engineering.py ← RFM metric computation
├── train_model.py       ← Scaling, Elbow Method, K-Means training
└── analysis.py          ← Cluster profiling & business labeling
```

---

## Methodology

### 1. Feature Engineering — RFM

Three behavioral signals are extracted per customer from raw transactional records:

| Metric | Description | Aggregation |
|---|---|---|
| **Recency** | Days since last purchase | `max(order_date)` → delta from reference date |
| **Frequency** | Number of unique orders | `nunique(order_id)` |
| **Monetary** | Total amount spent (R$) | `sum(price)` |

> **Key design decision:** The reference date is set to `max(purchase_date) + 1 day` across the entire dataset — not `today()`. This prevents data leakage: the Olist dataset ends in 2018, and using the system clock would make every customer appear inactive for years.

### 2. Preprocessing

RFM features are standardized with `StandardScaler` (zero mean, unit variance) before clustering. Without this step, the monetary dimension (R$ hundreds) would dominate recency (days) and distort cluster formation.

### 3. Optimal K — Elbow Method

A K-Means loop from K=1 to K=10 plots inertia (within-cluster sum of squares) against number of clusters. The "elbow" at **K=3** was selected as the inflection point for diminishing returns.

![Elbow Method Chart](elbow_method.png)

### 4. Clustering — K-Means

Final model trained with `n_clusters=3` and `random_state=42` for reproducibility.

---

## Results

```
Cluster Profiles (mean values per segment):

           recency    frequency    monetary
Regular    129 days      1.0       R$ 111
At Risk    389 days      1.0       R$ 111
VIP        241 days      1.0     R$ 1,025
```

```
Customer Distribution:

Regular     53,451   (55.8%)
At Risk     39,595   (41.3%)
VIP          2,786    (2.9%)
```

**Key finding:** VIP customers represent less than 3% of the base but spend ~10x more per transaction. A targeted retention strategy for this group has asymmetric revenue impact.

---

## How to Run

### Prerequisites

```bash
pip install pandas scikit-learn matplotlib
```

### Dataset

Download the [Olist dataset from Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place the following files in a `datasets/` folder:

```
datasets/
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
└── olist_order_reviews_dataset.csv
```

### Execute

```bash
python main.py
```

The pipeline will output cluster profiles to the terminal and save the Elbow Method chart as `elbow_method.png`.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` | Data manipulation and feature aggregation |
| `scikit-learn` | StandardScaler, KMeans |
| `matplotlib` | Elbow Method visualization |

---

## Project Structure

```
.
├── main.py                  # Entry point
├── data_prep.py             # ETL layer
├── feature_engineering.py  # RFM computation
├── train_model.py           # ML pipeline
├── analysis.py              # Business interpretation
├── elbow_method.png         # Cluster selection chart
└── datasets/                # Raw data (not versioned)
```