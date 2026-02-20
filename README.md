# Anomaly Detection in Ukrainian Public Procurement

Analyzing 12.9M ProZorro tenders (2022–2025) for suspicious patterns using a 5-method ensemble: rules, statistics, Isolation Forest, LOF, and network analysis.

## Structure

| Folder | Description |
|--------|-------------|
| `notebooks/` | Jupyter notebooks for EDA and analysis |
| `src/` | Python source code (detectors, data loaders) |
| `results/` | Figures and exported anomalies |
| `data/` | Dataset (download from Kaggle) |

## Dataset

**Source:** [ProZorro Ukraine Procurement 2022-2025](https://www.kaggle.com/datasets/romankachmar/prozorro-ukraine-procurement-2022-2025)

| File | Records |
|------|---------|
| `tenders_2022.csv` | 2.4M |
| `tenders_2023.csv` | 3.6M |
| `tenders_2024.csv` | 3.4M |
| `tenders_2025.csv` | 3.7M |
| `buyers.csv` | 36K |
| `suppliers.csv` | 359K |

Download and place in `data/` folder.

## Quick Start

```python
from src.data_loader import load_tenders, load_buyers
from src.detectors.rule_based import RuleBasedDetector

# Load data
tenders = load_tenders(years=[2023, 2024])
buyers = load_buyers()

# Run detection
detector = RuleBasedDetector()
results = detector.detect(tenders, buyers_df=buyers)

# Get high-risk tenders
high_risk = detector.get_high_risk()
```

## Methods

### Rule-Based Detection
- 45 red flag rules across 6 categories
- Risk scoring (critical/high/medium/low)
- Based on OCP methodology and ProZorro domain expertise

### ML Methods
- Isolation Forest (global anomalies, tender + aggregated level)
- LOF (local anomalies, aggregated level)
- Network Analysis (cartels, bid rotation, monopolistic pairs)

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Author

Roman Kachmar - Master's Thesis, 2026
