# Master Thesis - Anomaly Detection in Public Procurement

## Project Overview

**Topic:** Anomaly Detection in Ukrainian Public Procurement (ProZorro)
**Goal:** Develop ML models to detect suspicious patterns and potential fraud in 13M+ government tenders

## Project Structure

```
master-thesis/
├── thesis/              # Academic documents (PDF, DOCX)
├── data/                # Dataset (reference to prozorro-parser output)
├── notebooks/           # Jupyter notebooks for analysis
├── src/                 # Source code for models
├── results/             # Experiment results and figures
└── prozorro-parser/     # Dataset parser (separate project)
```

## Data Location

Dataset is in `data/` folder (~5.5 GB):
- `tenders_2022-2025.csv` - 13.1M tenders (55 columns each)
- `bids_2022-2025.csv` - 2.6M bids
- `buyers.csv` - 37K buyers (16 columns)
- `suppliers.csv` - 359K suppliers (6 columns)
- `bidders.csv` - 2.6M bidders (8 columns)

## Key Features for Anomaly Detection

### Competition Indicators
- `is_single_bidder` - only one participant
- `is_competitive` - more than one bidder
- `number_of_tenderers` - count of participants

### Price Indicators
- `price_change_pct` - discount/markup percentage
- `tender_value` vs `award_value` - price difference
- `discount_percentage_avg/max` - aggregated discounts

### Procurement Method
- `limited` (91%) - direct contracts, no competition
- `open` (5.5%) - competitive tenders
- `selective` (3.3%) - invited suppliers

### Red Flags
- `is_buyer_masked` / `is_supplier_masked` - hidden identities
- `has_unsuccessful_awards` / `has_cancelled_awards`
- `is_weekend` - unusual timing
- Buyer-supplier concentration (from buyers.csv)

## Research Questions

1. Can we detect anomalous pricing patterns?
2. Which buyer-supplier relationships are suspicious?
3. How did procurement patterns change during war?
4. What features best predict fraudulent tenders?

## Methodology

1. **EDA** - Exploratory analysis of patterns
2. **Feature Engineering** - Create fraud indicators
3. **Unsupervised Learning** - Isolation Forest, LOF, DBSCAN
4. **Supervised Learning** - If labeled data available
5. **Evaluation** - Domain expert validation

## Related Projects

- **prozorro-parser/** - Dataset creation and parsing
  - Location: `C:\Users\kachm\OneDrive\Робочий стіл\prozorro-parser`
  - Contains: Parser code, EDA notebooks, documentation

## Current Status

- [x] Dataset parsed (13.1M tenders)
- [x] Basic EDA completed
- [ ] Feature engineering
- [ ] Baseline models
- [ ] Anomaly detection models
- [ ] Thesis writing

## Tech Stack

- Python 3.11+
- pandas, numpy - data manipulation
- scikit-learn - ML models
- matplotlib, seaborn - visualization
- jupyter - notebooks

---
Last updated: 2026-01-10
