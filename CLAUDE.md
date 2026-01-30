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
│   ├── 02_rule_based.ipynb         # Level 1: Rule-based
│   ├── 03_statistical_screens.ipynb # Level 2: Statistical
│   ├── 04_isolation_forest.ipynb   # Level 3: ML (IF)
│   ├── 05_hdbscan.ipynb            # Level 3: ML (HDBSCAN + Cartel clusters)
│   ├── 06_ensemble.ipynb           # Cross-method validation
│   └── 07_network_analysis.ipynb   # Level 4: Network Analysis
├── src/                 # Source code for models
│   ├── config.py        # Thresholds, paths, constants
│   ├── data_loader.py   # Optimized data loading (13M+ records)
│   ├── features/        # Feature engineering (TODO)
│   ├── detectors/
│   │   ├── rule_based.py   # 44 red flag rules (DONE)
│   │   └── statistical.py  # Statistical screens (DONE)
│   └── evaluation/      # Cross-method validation (TODO)
├── results/             # Experiment results and figures
└── references/          # Research papers
```

## Data Location

Dataset is in `data/` folder (~5.3 GB):

**Tenders (year-based):**
- `tenders_2022.csv` - 2.4M tenders
- `tenders_2023.csv` - 3.6M tenders
- `tenders_2024.csv` - 3.4M tenders
- `tenders_2025.csv` - 3.7M tenders

**Bids (year-based):**
- `bids_2022.csv` - 242K bids
- `bids_2023.csv` - 489K bids
- `bids_2024.csv` - 847K bids
- `bids_2025.csv` - 1.1M bids

**Reference tables:**
- `buyers.csv` - 36K buyers (16 columns)
- `suppliers.csv` - 359K suppliers (6 columns)
- `bidders.csv` - 72K bidders (7 columns)

## Key Concept: Information Portrait

Інформаційний портрет - це вектор ~100 кількісних ознак, які описують суб'єкта закупівель:

- **Активність** - тенденції, обсяги, частота участі
- **Ризикові індикатори** - патерни, не окремі випадки
- **Темпоральна динаміка** - зміни поведінки у часі
- **Контекст** - норма для категорії CPV та регіону
- **Кореляційні патерни** - між індикаторами

**Новизна:** Перехід від статичних "червоних прапорців" до динамічного профілювання поведінки.

**buyers.csv вже містить початок портрету:**
- `single_bidder_rate`, `competitive_rate` - ризики
- `supplier_diversity_index` - патерн поведінки
- `avg_discount_pct`, `total_savings` - ефективність
- `first/last_tender_date` - темпоральність

## Anomaly Detection Methods

### Rule-based (DONE - `src/detectors/rule_based.py`)

**44 правила** в 6 категоріях (37 активних):

| Категорія | Правил | Приклади |
|-----------|--------|----------|
| Process Quality | 3 | R005 missing docs, R013 high limited usage |
| Competition Quality | 5 | R018 single bidder, R040 buyer-supplier dominance |
| Price Quality | 11 | R028 identical bids, R053 co-bidding same winner |
| Procedure Manipulation | 13 | R011 contract splitting, R002 threshold manipulation |
| Reputation | 2 | R048 heterogeneous supplier, R069 price increase |
| Additional | 10 | X009 single bidder low discount, X010 same day awards |

**Результати (2023, 10% sample):**
- Critical: ~2,100 (0.6%)
- High Risk: ~12,400 (3.5%)

**Використання:**
```python
from src.detectors.rule_based import RuleBasedDetector
detector = RuleBasedDetector()
results = detector.detect(tenders, bids_df=bids)
print(detector.summary())
```

### Statistical Screens (DONE - `src/detectors/statistical.py`)

**Статистичні тести** для конкурентних закупівель (Open/Selective з 3+ учасниками):

| Метод | Опис | Застосування |
|-------|------|--------------|
| Z-score | Виявляє викиди за стандартним відхиленням | Per-tender (ціни) |
| IQR | Interquartile range outliers | Per-tender |
| Benford's Law | Перевірка розподілу перших цифр | **Per-buyer, Per-supplier** (потребує 30+ зразків) |
| Bid Spread | CV, Min-Max різниця цін | Лише competitive (3+ bidders) |
| HHI | Herfindahl-Hirschman Index | Концентрація ринку |

**Важливо:** Статистичні тести (bid spread, award ratio) застосовуються ЛИШЕ до:
- Процедур Open або Selective
- Тендерів з 3+ учасниками

**Використання:**
```python
from src.detectors.statistical import StatisticalDetector
detector = StatisticalDetector()
results = detector.detect(tenders, bids_df=bids)
print(detector.summary())
```

### Machine Learning (unsupervised)
| Метод | Опис | Коли використовувати |
|-------|------|---------------------|
| **Isolation Forest** | Ізолює аномалії випадковими деревами | **Перший вибір** для табличних даних |
| **LOF** | Порівнює щільність точки з сусідами | Локальні аномалії |
| **One-Class SVM** | Границя навколо "нормальних" даних | Менші датасети |
| **DBSCAN** | Кластеризація, аномалії = шум | Пошук груп (картелі) |
| **Autoencoder** | Нейронка, аномалії = великий reconstruction error | Складні залежності |

### Network Analysis (DONE - `notebooks/07_network_analysis.ipynb`)

**Типи графів:**
| Граф | Вузли | Ребра | Що виявляє |
|------|-------|-------|------------|
| Bidder Co-participation | bidder_id | Спільна участь | Картелі |
| Winner-Loser | bidder_id | Переможець → Програвший | Bid-rigging rings |
| Buyer-Supplier | buyer + supplier | Контракт | Монопольні відносини |

**Метрики:**
- Community detection (Louvain) - групи тісно пов'язаних учасників
- Betweenness centrality - координатори змов
- Clustering coefficient - замкнуті групи
- Degree distribution - аномально активні учасники

### Рекомендований pipeline (з methodology_plan.pdf)
```
Level 1: Rule-based ✓ → Level 2: Statistical ✓ → Level 3: ML (IF, HDBSCAN) ✓ → Level 4: Network Analysis ✓ → Ensemble
```

## Key Features for Models

### З tenders:
- `is_single_bidder`, `is_competitive`, `number_of_tenderers`
- `price_change_pct`, `tender_value`, `award_value`
- `is_weekend`, `is_q4`, `is_december`
- `procurement_method` (limited/open/selective)

### З buyers (приєднати):
- `single_bidder_rate`, `competitive_rate`
- `supplier_diversity_index`
- `avg_discount_pct`

### З suppliers (приєднати):
- `total_awards`, `total_value`

## Procurement Methods Distribution

| Метод | % | Опис |
|-------|---|------|
| Limited | 91% | Прямі договори без торгів |
| Open | 5.5% | Конкурентні торги |
| Selective | 3.3% | Запрошені постачальники (зростає) |

## Current Status

- [x] Dataset parsed (13.1M tenders)
- [x] Basic EDA completed (`notebooks/01_eda.ipynb`)
- [x] Kaggle dataset uploaded
- [x] Buyers "portrait" features (16 columns)
- [x] Project structure (`src/`)
- [x] Data loader with memory optimization
- [x] **Level 1: Rule-based detector (44 rules, 37 active)**
- [x] **Level 2: Statistical screens (Benford, Z-score, HHI, Bid Spread)**
- [x] **Level 3: Isolation Forest (19 features, 5% contamination)**
- [x] **Level 3: HDBSCAN (clustering + outlier detection)**
- [x] **Level 4: Network Analysis (co-bidding, winner-loser, buyer-supplier graphs)**
- [x] **Ensemble notebook (cross-method validation)**
- [ ] Thesis writing

## Related Projects

- **prozorro-parser/** - Dataset creation and parsing
  - Location: `C:\Users\kachm\OneDrive\Робочий стіл\prozorro-parser`
  - Kaggle: https://www.kaggle.com/datasets/romankachmar/prozorro-ukraine-procurement-2022-2025

## Tech Stack

- Python 3.11+
- pandas, numpy - data manipulation
- scikit-learn - ML models (Isolation Forest, LOF, DBSCAN)
- matplotlib, seaborn - visualization
- jupyter - notebooks

---
Last updated: 2026-01-30
