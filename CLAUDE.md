# Master Thesis - Anomaly Detection in Public Procurement

## Project Overview

**Тема:** Метод формування інформаційних портретів суб'єктів публічних закупівель для інтелектуального аналізу та виявлення аномальних патернів

**Мета:** Розробка та впровадження методики формування інформаційних портретів суб'єктів публічних закупівель та їх аналізу методами машинного навчання без учителя для виявлення аномальних поведінкових патернів.

**Об'єкт:** Цифрове відображення діяльності суб'єктів публічних закупівель у системі Prozorro

**Предмет:** Інформаційні портрети суб'єктів публічних закупівель та методи ML для їх аналізу

**Advisor:** Сирота Олена

## Core Concept: Інформаційний портрет

**Інформаційний портрет** — це вектор з N агрегованих ознак, що характеризує поведінку суб'єкта закупівель (замовника, постачальника, або пари "замовник-постачальник").

```
Портрет = [feature_1, feature_2, ..., feature_N]
```

**Приклад портрета замовника (buyer, 12 features):**
```python
portrait = {
    'single_bidder_rate': 0.45,      # Частка закупівель з 1 учасником
    'competitive_rate': 0.55,        # Частка конкурентних процедур
    'avg_discount_pct': 12.3,        # Середня знижка від початкової ціни
    'supplier_diversity_index': 0.7, # Диверсифікація постачальників
    'total_tenders': 150,            # Кількість закупівель
    'total_value': 5_000_000,        # Загальний обсяг (грн)
    'avg_value': 33_333,             # Середня вартість закупівлі
    'cpv_concentration': 0.35,       # HHI по CPV категоріях (0-1)
    'avg_award_days': 45.2,          # Середній час до підписання (дні)
    'weekend_rate': 0.08,            # Частка закупівель у вихідні
    'value_variance_coeff': 1.5,     # Коефіцієнт варіації вартості
    'q4_rate': 0.30,                 # Частка закупівель у Q4
}
```

**Три рівні портретів:**
1. **Buyer-level** (~36K) — систематична поведінка замовника
2. **Supplier-level** (~360K) — патерни перемог постачальника
3. **Pair-level** (~916K) — взаємодія конкретної пари

**Ключова перевага:** Перехід від аналізу окремих тендерів до аналізу поведінкових патернів суб'єктів дозволяє виявляти систематичні (не разові) відхилення.

**Методи аналізу портретів:**
- **Isolation Forest** — глобальні аномалії (PyOD)
- **LOF** — локальні аномалії (PyOD)

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
│   ├── 06_aggregated_pyod.ipynb       # Aggregated-level IForest + LOF (buyer/supplier/pair)
│   ├── 07_final_results.ipynb         # Cross-method consensus analysis
│   └── 08_synthetic_validation.ipynb  # Synthetic anomaly injection validation
├── src/                 # Source code
│   ├── config.py        # Thresholds, paths, constants
│   ├── data_loader.py   # Polars-based data loading (FAST!)
│   └── detectors/
│       ├── __init__.py
│       ├── rule_based.py      # 45 red flag rules
│       ├── statistical.py     # Statistical screens
│       ├── pyod_detector.py   # PyOD: IForest (tender + aggregated) + LOF (aggregated only)
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

**45 правил** в 6 категоріях:

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

### Level 3: ML - Isolation Forest (PyOD)

**Two levels of analysis** — IForest is the only ML method that works at both tender-level and aggregated level:

#### Tender-level (`PyODDetector`) — 12.9M tenders, 9 features incl. `value_vs_cpv_median`:

```python
from src.detectors import PyODDetector

detector = PyODDetector(algorithm="iforest", contamination=0.05)
results = detector.fit_detect(tenders)

# Feature importance
fi = detector.feature_importances()  # Returns sorted dict
```

#### Aggregated-level (`AggregatedPyOD`):

```python
from src.detectors import AggregatedPyOD

detector = AggregatedPyOD(algorithm="iforest", contamination=0.05)

# Three levels of analysis (fresh aggregation, no buyers_df needed)
buyer_results = detector.detect_buyers(tenders)
supplier_results = detector.detect_suppliers(tenders)
pair_results = detector.detect_pairs(tenders, min_contracts=3)

# Get anomalies
suspicious = detector.get_anomalies("buyers", min_score=0.5)

# Feature importance (IForest — computed from 100 trees)
fi = detector.feature_importances("buyers")  # Returns sorted dict
```

### Level 3: ML - LOF (Local Outlier Factor)

LOF detects contextual anomalies — entities that are abnormal relative to their neighbors. Complementary to IForest (Jaccard = 0.032 on buyer-level, 0.005 on supplier-level). **Aggregated-level only** — O(n²) complexity makes tender-level (12.9M records) impossible; works on ~36K buyers in ~8s (12 features).

```python
from src.detectors import AggregatedPyOD

detector = AggregatedPyOD(algorithm="lof", contamination=0.05)

# Three levels of analysis
buyer_results = detector.detect_buyers(tenders)
supplier_results = detector.detect_suppliers(tenders)
pair_results = detector.detect_pairs(tenders, min_contracts=3)

# Get anomalies
suspicious = detector.get_anomalies("buyers", min_score=0.5)
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

Combine multiple methods with explanations:

```python
from src.detectors import EnsembleDetector

detector = EnsembleDetector(
    weights={"rule": 1.0, "stat": 0.8, "if": 1.0, "lof": 0.8, "network": 1.0},
    consensus_threshold=2,
)
results = detector.combine(
    rule_results=rule_df, stat_results=stat_df,
    if_results=if_df, lof_results=lof_df,
    network_results=network_df,
)

# Generate human-readable explanations for flagged tenders
explanations = detector.generate_explanations(
    tenders_df=tenders, rule_results=rule_df,
    network_results=network_df, buyer_portraits=buyers,
)
```

## Preprocessing Pipeline

All ML detectors use the same preprocessing:

```
Raw Features → Log-transform (skewed) → Impute (median) → RobustScale → Model
```

**Log-transformed features** (monetary + counts):
- `total_value`, `tender_value`, `award_value`, `avg_value`
- `total_awards`, `total_tenders`, `contracts_count`, `buyer_count`
- `value_vs_cpv_median` (ratio of tender value to CPV category median)
- `avg_award_days`, `cpv_diversity`

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
- `value_vs_cpv_median` (award_value / median for same CPV 2-digit category)

### Buyer-level (12 features from aggregation):
- `single_bidder_rate`, `competitive_rate`, `avg_discount_pct`
- `supplier_diversity_index`, `total_tenders`, `total_value`, `avg_value`
- `cpv_concentration` (HHI of CPV categories), `avg_award_days`, `weekend_rate`
- `value_variance_coeff` (CV of tender values), `q4_rate`

### Supplier-level (7 features):
- `total_awards`, `total_value`, `avg_award_value`
- `buyer_count`, `single_bidder_rate`, `avg_competitors`
- `cpv_diversity` (number of unique CPV categories)

### Pair-level (7 features):
- `contracts_count`, `total_value`, `avg_value`
- `single_bidder_rate`, `exclusivity_buyer`, `exclusivity_supplier`
- `temporal_concentration` (std of days between contracts)

## Current Status

### Code & Experiments
- [x] Dataset parsed (12.9M tenders)
- [x] Polars data loader (10-100x faster)
- [x] **Level 1:** Rule-based detector (45 rules, incl. X011 high-value limited)
- [x] **Level 2:** Statistical screens (Benford, Z-score, HHI)
- [x] **Level 3:** Isolation Forest via PyOD (tender-level + aggregated)
- [x] **Level 3:** LOF via PyOD (aggregated, core ensemble method)
- [x] **Level 4:** Network Analysis (configurable thresholds)
- [x] **Ensemble:** Cross-method validation with explanations
- [x] Log-transform preprocessing
- [x] `value_vs_cpv_median` feature for IForest (value contextual to CPV category)
- [x] Human-readable explanation column for critical tenders
- [x] Extended portraits: buyer (12 features), supplier (7), pair (7)
- [x] Feature importance for IForest (computed from individual trees)

### Thesis Writing (структура: Вступ + 2 розділи + Висновки)
- [x] Розділ 1 / Вступ (intro_updated.md) — актуальність, мета, об'єкт, предмет, методи
- [x] Розділ 2 / Теоретико-методологічні засади (thesis_draft.pdf, стор. 8-22) — огляд літератури, підходи, концепція портретів
- [x] Розділ 3 / Експериментальне дослідження:
  - [x] 3.1 (chapter3_portfolio.md) — методологія оцінки результатів
  - [x] 3.2 — збір та обробка даних
  - [x] 3.3 (section_3_3.md) — формування портретів (формальне визначення)
  - [x] 3.4 (section_3_4.md) — реалізація та результати методів
  - [x] 3.5 (section_3_5.md) — ансамблевий аналіз та крос-валідація
  - [x] 3.6 (section_3_6.md) — валідація результатів + висновки до розділу
- [x] Розділ 2 емпіричні числа (chapter4_experiments.md)
- [ ] Висновки (загальні)

### Output Files
- `results/critical_tenders.csv` — Critical tenders with explanation column + buyer/supplier details
- `results/ensemble_summary.csv` — Summary statistics
- `results/network_*.csv` — Network analysis results (bid rotation, monopolistic, communities)
- `results/synthetic_validation_*.csv` — Validation results

## Tech Stack

- **Python 3.11+**
- **Polars** - Fast data loading and aggregation
- **Pandas/NumPy** - ML compatibility
- **scikit-learn** - Preprocessing, metrics
- **PyOD** - Anomaly detection (IForest + LOF)
- **igraph** - Fast graph analysis (community detection, centrality)
- **NetworkX** - Graph utilities and visualization
- **Matplotlib/Seaborn** - Visualization

## Related Projects

- **prozorro-parser/** - Dataset creation
  - Location: `C:\Users\kachm\OneDrive\Робочий стіл\prozorro-parser`
  - Kaggle: https://www.kaggle.com/datasets/romankachmar/prozorro-ukraine-procurement-2022-2025

---
Last updated: 2026-02-10
