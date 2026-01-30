# Master Thesis - Anomaly Detection in Public Procurement

## Project Overview

**Topic:** Дослідження методів аналізу аномалій у державних закупівлях з використанням машинного навчання

**Goal:** Develop ML models to detect suspicious patterns and potential fraud in 13M+ government tenders

**Advisor:** Сирота Олена

## Project Structure

```
master-thesis/
├── thesis/              # Academic documents (PDF, references)
├── data/                # Dataset (from prozorro-parser)
├── notebooks/           # Jupyter notebooks for analysis
│   ├── 01_eda.ipynb
│   ├── 02_rule_based.ipynb            # Level 1: Rule-based
│   ├── 03_statistical_screens.ipynb   # Level 2: Statistical
│   ├── 04_isolation_forest.ipynb      # Level 3: ML (IF)
│   ├── 05_hdbscan.ipynb               # Level 3: ML (HDBSCAN)
│   ├── 06_ensemble.ipynb              # Cross-method validation
│   ├── 07_network_analysis.ipynb      # Level 4: Network Analysis
│   ├── 08_pyod_comparison.ipynb       # PyOD algorithms comparison
│   └── 09_aggregated_hdbscan.ipynb    # HDBSCAN on aggregated levels
├── src/                 # Source code
│   ├── config.py        # Thresholds, paths, constants
│   ├── data_loader.py   # Polars-based data loading (FAST!)
│   └── detectors/
│       ├── __init__.py
│       ├── rule_based.py      # 44 red flag rules
│       ├── statistical.py     # Statistical screens
│       ├── isolation_forest.py # Isolation Forest
│       ├── hdbscan.py         # HDBSCAN + AggregatedHDBSCAN
│       ├── pyod_detector.py   # PyOD unified interface (7 algorithms)
│       ├── network.py         # Network/Graph analysis
│       └── ensemble.py        # Ensemble detector
├── results/             # Experiment results and figures
└── references/          # Research papers
```

## Data Loading (Polars)

Data loader uses **Polars** for 10-100x faster loading:

```python
from src.data_loader import load_tenders, load_buyers, load_suppliers

# Load data (returns Pandas by default)
tenders = load_tenders(years=[2023], sample_frac=0.1)
buyers = load_buyers()

# Return Polars DataFrame for faster operations
tenders_pl = load_tenders(years=[2023], return_polars=True)

# Fast aggregations (Polars-native)
from src.data_loader import aggregate_by_buyer, aggregate_by_supplier, aggregate_by_pair
buyer_agg = aggregate_by_buyer(tenders)      # Buyer-level features
supplier_agg = aggregate_by_supplier(tenders) # Supplier-level features
pair_agg = aggregate_by_pair(tenders)         # Buyer-supplier pairs
```

## Detectors Overview

### Level 1: Rule-based (`RuleBasedDetector`)

**44 правила** в 6 категоріях:

```python
from src.detectors import RuleBasedDetector

detector = RuleBasedDetector()
results = detector.detect(tenders, bids_df=bids)
print(detector.summary())
```

### Level 2: Statistical (`StatisticalDetector`)

Benford's Law, Z-score, IQR, HHI, Bid Spread:

```python
from src.detectors import StatisticalDetector

detector = StatisticalDetector()
results = detector.detect(tenders, bids_df=bids)
```

### Level 3: ML - PyOD (RECOMMENDED)

**Unified interface for 7 algorithms:**

| Algorithm | Type | Speed | Description |
|-----------|------|-------|-------------|
| `iforest` | Tree-based | Fast | Isolation Forest (default) |
| `lof` | Density | Medium | Local Outlier Factor |
| `knn` | Distance | Medium | K-Nearest Neighbors |
| `hbos` | Histogram | **Fastest** | Histogram-based |
| `ecod` | Distribution | Fast | Empirical CDF (parameter-free) |
| `copod` | Distribution | Fast | Copula-based (parameter-free) |
| `ocsvm` | Boundary | Slow | One-Class SVM |

```python
from src.detectors import PyODDetector, compare_algorithms

# Single algorithm
detector = PyODDetector(algorithm="iforest", contamination=0.05)
results = detector.fit_detect(tenders, buyers_df=buyers)

# Compare multiple algorithms
comparison = compare_algorithms(
    tenders,
    algorithms=["iforest", "hbos", "ecod", "lof"],
    contamination=0.05
)
```

### Level 3: ML - HDBSCAN

**Two modes:**

1. **Tender-level** - cluster individual tenders:
```python
from src.detectors import HDBSCANDetector

detector = HDBSCANDetector(min_cluster_size=50)
results = detector.fit_detect(tenders, buyers_df=buyers)
```

2. **Aggregated-level** (RECOMMENDED) - cluster buyers/suppliers/pairs:
```python
from src.detectors import AggregatedHDBSCAN

detector = AggregatedHDBSCAN(min_cluster_size=10)

# Three levels of analysis
buyer_results = detector.cluster_buyers(tenders, buyers)
supplier_results = detector.cluster_suppliers(tenders)
pair_results = detector.cluster_pairs(tenders, min_contracts=3)

# Get anomalies
suspicious_buyers = detector.get_suspicious_buyers(min_score=0.5)
suspicious_suppliers = detector.get_suspicious_suppliers(min_score=0.5)
suspicious_pairs = detector.get_suspicious_pairs(min_score=0.5)
```

### Level 4: Network Analysis (`NetworkAnalysisDetector`)

Graph-based detection with configurable thresholds:

```python
from src.detectors import NetworkAnalysisDetector

detector = NetworkAnalysisDetector(
    min_co_bids=3,
    min_contracts=3,
    # Anomaly thresholds (stricter = fewer flags)
    suspicious_min_degree=10,
    suspicious_min_clustering=0.7,
    rotation_min_ratio=0.7,
    monopoly_min_ratio=0.9,
    monopoly_min_contracts=20,
)

results = detector.detect(tenders, bids_df=bids)
```

### Ensemble (`EnsembleDetector`)

Combine multiple methods:

```python
from src.detectors import EnsembleDetector

detector = EnsembleDetector(methods=["rule", "statistical", "iforest"])
results = detector.fit_detect(tenders, bids_df=bids, buyers_df=buyers)
```

## Preprocessing Pipeline

All ML detectors use the same preprocessing:

```
Raw Features → Log-transform (skewed) → Impute (median) → RobustScale → Model
```

**Log-transformed features** (monetary + counts):
- `total_value`, `tender_value`, `award_value`, `avg_value`
- `total_awards`, `total_tenders`, `contracts_count`, `buyer_count`

## Data Location

Dataset in `data/` folder (~5.3 GB):

| File | Records | Description |
|------|---------|-------------|
| `tenders_2022.csv` | 2.4M | Tenders |
| `tenders_2023.csv` | 3.6M | Tenders |
| `tenders_2024.csv` | 3.4M | Tenders |
| `tenders_2025.csv` | 3.7M | Tenders |
| `bids_*.csv` | 2.7M total | Bid-level data |
| `buyers.csv` | 36K | Buyer profiles |
| `suppliers.csv` | 359K | Supplier profiles |

## Key Features

### Tender-level:
- `tender_value`, `award_value`, `price_change_pct`
- `is_single_bidder`, `is_competitive`, `number_of_tenderers`
- `is_weekend`, `is_q4`, `is_december`

### Buyer-level (from `buyers.csv` or aggregation):
- `single_bidder_rate`, `competitive_rate`
- `supplier_diversity_index`
- `avg_discount_pct`, `total_value`

### Supplier-level:
- `total_awards`, `total_value`
- `buyer_count`, `single_bidder_rate`

### Pair-level (buyer-supplier):
- `contracts_count`, `exclusivity_buyer`, `exclusivity_supplier`

## Current Status

- [x] Dataset parsed (13.1M tenders)
- [x] Polars data loader (10-100x faster)
- [x] **Level 1:** Rule-based detector (44 rules)
- [x] **Level 2:** Statistical screens (Benford, Z-score, HHI)
- [x] **Level 3:** PyOD detector (7 algorithms)
- [x] **Level 3:** HDBSCAN (tender + aggregated levels)
- [x] **Level 4:** Network Analysis (configurable thresholds)
- [x] **Ensemble:** Cross-method validation
- [x] Log-transform preprocessing
- [ ] Thesis writing

## Tech Stack

- **Python 3.11+**
- **Polars** - Fast data loading and aggregation
- **Pandas/NumPy** - ML compatibility
- **scikit-learn** - Preprocessing, metrics
- **PyOD** - Anomaly detection algorithms
- **HDBSCAN** - Clustering
- **NetworkX** - Graph analysis
- **Matplotlib/Seaborn** - Visualization

## Related Projects

- **prozorro-parser/** - Dataset creation
  - Location: `C:\Users\kachm\OneDrive\Робочий стіл\prozorro-parser`
  - Kaggle: https://www.kaggle.com/datasets/romankachmar/prozorro-ukraine-procurement-2022-2025

---
Last updated: 2026-01-30
