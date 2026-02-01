# Master Thesis - Anomaly Detection in Public Procurement

## Project Overview

**Тема:** Інформаційні портрети суб'єктів публічних закупівель: формування, аналіз та виявлення аномалій методами машинного навчання

**Мета:** Розробка та впровадження методики формування інформаційних портретів суб'єктів публічних закупівель та їх аналізу методами машинного навчання без учителя для виявлення аномальних поведінкових патернів.

**Об'єкт:** Цифрове відображення діяльності суб'єктів публічних закупівель у системі Prozorro

**Предмет:** Інформаційні портрети суб'єктів публічних закупівель та методи ML для їх аналізу

**Advisor:** Сирота Олена

## Core Concept: Інформаційний портрет

**Інформаційний портрет** — це вектор з N агрегованих ознак, що характеризує поведінку суб'єкта закупівель (замовника, постачальника, або пари "замовник-постачальник").

```
Портрет = [feature_1, feature_2, ..., feature_N]
```

**Приклад портрета замовника (buyer):**
```python
portrait = {
    'single_bidder_rate': 0.45,      # Частка закупівель з 1 учасником
    'competitive_rate': 0.55,        # Частка конкурентних процедур
    'avg_discount_pct': 12.3,        # Середня знижка від початкової ціни
    'supplier_diversity_index': 0.7, # Диверсифікація постачальників
    'total_tenders': 150,            # Кількість закупівель
    'total_value': 5_000_000,        # Загальний обсяг (грн)
    'avg_tender_value': 33_333,      # Середня вартість закупівлі
}
```

**Три рівні портретів:**
1. **Buyer-level** (~36K) — систематична поведінка замовника
2. **Supplier-level** (~360K) — патерни перемог постачальника
3. **Pair-level** (~500K) — взаємодія конкретної пари

**Ключова перевага:** Перехід від аналізу окремих тендерів до аналізу поведінкових патернів суб'єктів дозволяє виявляти систематичні (не разові) відхилення.

**Методи аналізу портретів:**
- **Isolation Forest** — глобальні аномалії
- **LOF** — локальні аномалії (нетиповий для свого контексту)
- **HDBSCAN** — кластеризація + виявлення outliers
- **Autoencoder** — складні нелінійні патерни

**Thesis documents:** `thesis/intro_updated.md`, `thesis/chapter3_portfolio.md`, `thesis/chapter4_experiments.md`

## Project Structure

```
master-thesis/
├── thesis/              # Academic documents (PDF, references)
├── data/                # Dataset (from prozorro-parser)
├── notebooks/           # Jupyter notebooks for analysis
│   ├── 01_eda.ipynb
│   ├── 02_rule_based.ipynb            # Level 1: Rule-based
│   ├── 03_statistical_screens.ipynb   # Level 2: Statistical
│   ├── 04_ensemble.ipynb              # Cross-method validation
│   ├── 05_network_analysis.ipynb      # Network/Graph analysis
│   ├── 06_pyod_comparison.ipynb       # Tender-level PyOD (fast algorithms)
│   ├── 07_aggregated_hdbscan.ipynb    # Aggregated HDBSCAN clustering
│   └── 08_aggregated_pyod.ipynb       # Buyer-level PyOD (KNN, LOF, OCSVM)
├── src/                 # Source code
│   ├── config.py        # Thresholds, paths, constants
│   ├── data_loader.py   # Polars-based data loading (FAST!)
│   └── detectors/
│       ├── __init__.py
│       ├── rule_based.py      # 44 red flag rules
│       ├── statistical.py     # Statistical screens
│       ├── pyod_detector.py   # PyOD: tender-level (fast) + aggregated (+ KNN, LOF, OCSVM)
│       ├── hdbscan.py         # HDBSCANDetector + AggregatedHDBSCAN
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

**Two levels of analysis:**

#### Tender-level (`PyODDetector`) - 6 fast algorithms:

| Algorithm | Type | Speed | Description |
|-----------|------|-------|-------------|
| `iforest` | Tree-based | Fast | Isolation Forest (default) |
| `hbos` | Histogram | **Fastest** | Histogram-based |
| `ecod` | Distribution | Fast | Empirical CDF (parameter-free) |
| `copod` | Distribution | Fast | Copula-based (parameter-free) |
| `autoencoder` | Neural | Medium | AutoEncoder (reconstruction error) |
| `vae` | Neural | Medium | Variational AutoEncoder |

```python
from src.detectors import PyODDetector, compare_algorithms

# Single algorithm
detector = PyODDetector(algorithm="iforest", contamination=0.05)
results = detector.fit_detect(tenders, buyers_df=buyers)

# Compare algorithms
comparison = compare_algorithms(tenders, algorithms=["iforest", "hbos", "ecod"])
```

#### Aggregated-level (`AggregatedPyOD`) - 9 algorithms (+ O(n²)):

KNN, LOF, OCSVM доступні тільки на агрегованому рівні, бо O(n²) — повільні на 13M тендерів, але швидкі на 36K buyers.

```python
from src.detectors import AggregatedPyOD

# LOF on aggregated data (recommended)
detector = AggregatedPyOD(algorithm="lof", contamination=0.05)

# Three levels of analysis
buyer_results = detector.detect_buyers(tenders, buyers)
supplier_results = detector.detect_suppliers(tenders)
pair_results = detector.detect_pairs(tenders, min_contracts=3)

# Get anomalies
suspicious = detector.get_anomalies("buyers", min_score=0.5)
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

### Level 3: ML - AutoEncoder / VAE (Deep Learning via PyOD)

Uses reconstruction error as anomaly score:

```python
from src.detectors import PyODDetector, AggregatedPyOD

# Tender-level AutoEncoder
detector = PyODDetector(algorithm="autoencoder", contamination=0.05)
results = detector.fit_detect(tenders, buyers_df=buyers)

# Tender-level VAE
detector = PyODDetector(algorithm="vae", contamination=0.05)
results = detector.fit_detect(tenders, buyers_df=buyers)

# Aggregated-level (recommended)
detector = AggregatedPyOD(algorithm="autoencoder", contamination=0.05)
buyer_results = detector.detect_buyers(tenders, buyers)
supplier_results = detector.detect_suppliers(tenders)
pair_results = detector.detect_pairs(tenders, min_contracts=3)
```

### Level 4: Network Analysis (`NetworkAnalysisDetector`)

Graph-based detection with 4 graph types:

| Graph | Nodes | Edges | Detects |
|-------|-------|-------|---------|
| Co-bidding | bidders | co-participation | Cartels |
| Winner-Loser | bidders | winner→loser | Bid rotation |
| Buyer-Supplier | buyers + suppliers | contracts | Monopoly |
| **Full Collusion** | **all combined** | **all edges** | **Complex schemes** |

```python
from src.detectors import NetworkAnalysisDetector

detector = NetworkAnalysisDetector(
    min_co_bids=3,
    min_contracts=3,
    # Anomaly thresholds
    suspicious_min_degree=10,
    suspicious_min_clustering=0.7,
    rotation_min_ratio=0.7,
    monopoly_min_ratio=0.9,
    monopoly_min_contracts=20,
)

results = detector.fit_detect(tenders, bids_df=bids)

# Get collusion communities (buyers + suppliers together)
collusion_communities = detector.get_collusion_communities(min_size=5)
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

### Code & Experiments
- [x] Dataset parsed (13.1M tenders)
- [x] Polars data loader (10-100x faster)
- [x] **Level 1:** Rule-based detector (44 rules)
- [x] **Level 2:** Statistical screens (Benford, Z-score, HHI)
- [x] **Level 3:** PyOD detector (6 tender-level + 7 aggregated with LOF)
- [x] **Level 3:** HDBSCAN (tender + aggregated levels)
- [x] **Level 3:** Autoencoder (deep learning, reconstruction error)
- [x] **Level 4:** Network Analysis (configurable thresholds)
- [x] **Ensemble:** Cross-method validation
- [x] Log-transform preprocessing

### Thesis Writing
- [x] Вступ (intro_updated.md) — тема, мета, об'єкт, предмет
- [x] Розділ 3 (chapter3_portfolio.md) — методика формування портретів
- [x] Розділ 4 (chapter4_experiments.md) — структура експериментів
- [ ] Розділ 1 — теоретичні засади
- [ ] Розділ 2 — огляд літератури
- [ ] Заповнення результатів експериментів
- [ ] Висновки

## Tech Stack

- **Python 3.11+**
- **Polars** - Fast data loading and aggregation
- **Pandas/NumPy** - ML compatibility
- **scikit-learn** - Preprocessing, metrics
- **PyOD** - Anomaly detection algorithms
- **HDBSCAN** - Clustering
- **TensorFlow/Keras** - Neural networks (PyOD AutoEncoder, VAE)
- **igraph** - Fast graph analysis (community detection, centrality)
- **NetworkX** - Graph utilities and visualization
- **Matplotlib/Seaborn** - Visualization

## Related Projects

- **prozorro-parser/** - Dataset creation
  - Location: `C:\Users\kachm\OneDrive\Робочий стіл\prozorro-parser`
  - Kaggle: https://www.kaggle.com/datasets/romankachmar/prozorro-ukraine-procurement-2022-2025

---
Last updated: 2026-01-31
